from genericpath import isfile
import os
import wandb
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from models import Diffusion, UNet
from utils import video_to_stream, save_video, save_img, get_temp_path, load_img


class LightningDiffusion(pl.LightningModule):
    def __init__(
        self, image_size, in_channels, model_channels, out_channels, 
        num_res_blocks, attention_resolutions,
        dropout, channel_mult, num_heads, num_head_channels, 
        resblock_updown, n_timesteps, id_condition_type, 
        audio_condition_type, lr, bsz, vlb_weight, lip_weight, precision, 
        video_rate, audio_rate, log_dir, n_motion_frames=0,
        motion_transforms=None, grayscale_motion=False, n_audio_motion_embs=0,
        n_epochs=None):
        super(LightningDiffusion, self).__init__()

        self.unet = UNet(
            image_size, in_channels, model_channels, out_channels,
            num_res_blocks, attention_resolutions,
            dropout=dropout, channel_mult=channel_mult, num_heads=num_heads,
            num_head_channels=num_head_channels, resblock_updown=resblock_updown, 
            id_condition_type=id_condition_type, audio_condition_type=audio_condition_type,
            precision=precision, n_motion_frames=n_motion_frames, 
            grayscale_motion=grayscale_motion, n_audio_motion_embs=n_audio_motion_embs
            )

        self.diffusion = Diffusion(
            self.unet, n_timesteps, in_channels, image_size, 
            out_channels, precision=precision,
            motion_transforms=motion_transforms
            )
        
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        if self.hparams.lip_weight:
            x, x_cond, motion_frames, audio_emb, landmarks = batch
        else:
            x, x_cond, motion_frames, audio_emb = batch
            landmarks = None

        x = x.to(self.device)
        x_cond = x_cond.to(self.device)
        motion_frames = motion_frames.to(self.device)

        losses = self.diffusion(x, x_cond, motion_frames=motion_frames, audio_emb=audio_emb, landmarks=landmarks)
        loss = losses['simple']
        if 'vlb' in losses:
            loss += losses['vlb'] * self.hparams.vlb_weight
            self.log('loss_vlb', losses['vlb'].item())
            self.log('loss_simple', losses['simple'].item())
        if 'lip' in losses:
            loss += losses['lip'] * self.hparams.lip_weight
            self.log('loss_lip', losses['lip'].item())
        self.log('loss', loss.item())

        # for last-K checkpoint saving
        self.log('monitoring_step', self.global_step)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            mode = 'test'
        else:
            mode = 'val'

        x_cond_stacked, audio_embs_padded, audio_list, seq_lens, file_names = batch

        samples = self.diffusion.sample(x_cond_stacked, self.hparams.bsz, audio_emb=audio_embs_padded, device=self.device, mode=mode)
        
        if x_cond_stacked.shape[0] == 1:
            self.save_media(audio_list[0], x_cond_stacked, samples, mode=mode, file_name=file_names[0])
        else:
            for audio, x_cond, sample, seq_len, file_name in zip(audio_list, x_cond_stacked, samples, seq_lens, file_names):
                self.save_media(audio, x_cond, sample[:seq_len], mode=mode, file_name=file_name)
    
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, 0)
        self.log('Sample batch id', batch_idx)

    def on_validation_epoch_end(self):
        self.log('lr', self.optimizers().optimizer.param_groups[0]['lr'], sync_dist=True)
        for mode in ['test', 'val']:
            self.log_media(mode, tmp_dir=self.hparams.log_dir)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        if self.hparams.n_epochs is not None:
            step_size = self.hparams.n_epochs // 18
            print(f'Scheduler {step_size}')
        else:
            step_size = 166
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.8)
        return [optimizer], [scheduler]

    def save_media(self, audio, driving_frame, video, mode='', file_name=None):
        temp_file_vid = os.path.join(self.hparams.log_dir, mode, file_name + '.mp4')
        save_video(temp_file_vid, video, audio=audio, fps=self.hparams.video_rate, audio_rate=self.hparams.audio_rate)

    @rank_zero_only
    def log_media(self, mode, tmp_dir='./tmp', keep_files=False):
        media_dir = os.path.join(tmp_dir, mode)
        if not os.path.exists(media_dir):
            os.makedirs(media_dir)

        file_list = []
        def traverse_dirs(path):
            if os.path.isfile(path):
                file_list.append(path)
            else:
                for subpath in os.listdir(path):
                    traverse_dirs(os.path.join(path, subpath))
        traverse_dirs(media_dir)

        for file in file_list:
            temp_file = os.path.join(media_dir, file)
            if temp_file.endswith('.mp4'):
                self.logger.experiment.log({f"Video {mode}": wandb.Video(video_to_stream(os.path.join(media_dir, temp_file), keep_files=keep_files), format="mp4")})
            elif temp_file.endswith('.png'):
                driving_frame = load_img(os.path.join(media_dir, temp_file))
                self.logger.log_image(key=f'Driving frame {mode}', images=[driving_frame])