import os
import yaml
import hydra
from datetime import datetime
from tqdm import trange
import wandb
from PIL import Image
import numpy as np
import torch
import torchaudio
from torchvision.transforms import ToTensor, Resize
import decord

from lightning_modules import LightningDiffusion

from utils import save_video, load_image_to_torch, get_temp_path


@hydra.main(config_path='./configs', config_name='config_gen_custom', version_base='1.1')
def main(args):
    # if not args.debug:
    #     wandb.login()
    #     wandb_logger = WandbLogger(
    #         project='', entity='', 
    #         name=f'Id + audio_emb CREMA {args.id_condition_type} {args.audio_condition_type} n_motion_frames {args.n_motion_frames} {datetime.now()}', 
    #         settings=wandb.Settings(start_method="fork")
    #         )

    device = 'cuda'
    torchaudio.set_audio_backend("sox_io")
    decord.bridge.set_bridge('torch')

    audio, audio_rate = torchaudio.load(args.audio_dir, channels_first=False)
    audio_emb = torch.load(args.audio_emb_dir).to(device)

    if args.id_frame_from_video:
        vr = decord.VideoReader(args.video_dir)
        id_frame = vr.get_batch([0]).permute(0, 3, 1, 2)
    else:
        id_frame = load_image_to_torch(args.id_frame_dir).unsqueeze(0).to(device)

    id_frame = (id_frame / 255) * 2 - 1
    id_frame = Resize((args.img_resize, args.img_resize))(id_frame).to(device)

    model = LightningDiffusion.load_from_checkpoint(args.checkpoint).to(device)
    model.diffusion.space(args.n_sample_timesteps)
    
    samples = model.diffusion.sample(id_frame, model.hparams.bsz, audio_emb=audio_emb.unsqueeze(0), device=model.device, mode='test', stabilize=args.stabilize, segment_len=args.segment_len)

    file_dir = get_temp_path(args.log_dir, ext='.mp4')
    save_video(file_dir, samples, audio=audio, fps=model.hparams.video_rate, audio_rate=audio_rate)


if __name__ == '__main__':
    main()