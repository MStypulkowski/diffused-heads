import os
import yaml
import hydra
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from lightning_modules import LightningDiffusion, DataModule
from utils import get_motion_transforms, get_exp_name, CustomModelCheckpoint


@hydra.main(config_path='./configs', config_name='config', version_base='1.1')
def main(args):
    wandb.login()
    wandb_logger = WandbLogger(
        project='talking-faces', entity='talking-faces', 
        name=get_exp_name(args), 
        settings=wandb.Settings(start_method="fork"),
        # offline=args.debug
        )

    if args.debug:
        args.n_timesteps = 10

    if args.lip_weight:
        landmarks = True
    else:
        landmarks = False
    
    print(args)

    pl.seed_everything(2137)

    motion_transforms = get_motion_transforms(args)

    data_module = DataModule(
        args.data_dir, args.bsz, args.img_resize, 
        n_workers=args.n_workers, identity_frame=args.identity_frame,
        audio_emb_dir=args.audio_emb_dir, n_motion_frames=args.n_motion_frames, 
        motion_transforms=motion_transforms, n_audio_motion_embs=args.n_audio_motion_embs,
        landmarks=landmarks, effective_bsz=args.bsz * torch.cuda.device_count() * args.n_nodes
        )
    
    image_size, in_channels = data_module.dataset_train[0][0].shape[1], data_module.dataset_train[0][0].shape[0]

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.checkpoint:
        model = LightningDiffusion.load_from_checkpoint(args.checkpoint)
    else:
        model = LightningDiffusion(
            image_size, in_channels, args.model_channels, args.out_channels, 
            args.num_res_blocks, args.attention_resolutions, args.dropout, 
            args.channel_mult, args.num_heads, args.num_head_channels, 
            args.resblock_updown, args.n_timesteps, args.id_condition_type,
            args.audio_condition_type, args.lr, args.bsz, args.vlb_weight, 
            args.lip_weight, args.precision, data_module.video_rate, 
            data_module.audio_rate, args.log_dir, n_motion_frames=args.n_motion_frames,
            motion_transforms=motion_transforms, grayscale_motion=args.grayscale_motion,
            n_audio_motion_embs=args.n_audio_motion_embs, n_epochs=args.n_epochs)
    
    print('\n', model.hparams, '\n')

    if args.n_epochs > 1000:
        every_n_epochs = 50
    elif args.data_dir.split('/')[-1] == 'avspeech':
        every_n_epochs = 1
    else:
        every_n_epochs = 2
    callbacks = [
            TQDMProgressBar(refresh_rate=10), 
            CustomModelCheckpoint(save_on_train_epoch_end=True, monitor='monitoring_step', save_top_k=5, mode='max', every_n_epochs=every_n_epochs)
            ]
    if args.swa_lr:
        callbacks.append(StochasticWeightAveraging(swa_lrs=args.swa_lr, swa_epoch_start=args.swa_epoch_start))

    if not args.debug:
        trainer = Trainer(
            precision=args.precision,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=-1,
            strategy=DDPStrategy(find_unused_parameters=False),
            num_nodes=args.n_nodes,
            max_epochs=args.n_epochs,
            callbacks=callbacks,
            logger=wandb_logger,
            log_every_n_steps=20,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=args.val_every_n_epochs,
            resume_from_checkpoint=args.checkpoint
        )
    else:
        trainer = Trainer(
            precision=args.precision,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=-1,
            strategy=DDPStrategy(find_unused_parameters=False),
            num_nodes=1,
            max_epochs=10,
            callbacks=callbacks,
            logger=wandb_logger,
            # profiler='simple',
            limit_train_batches=0.0001,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=1
        )

    trainer.fit(model, data_module)

    # Log remaining samples
    model.log_media('val', tmp_dir=args.log_dir)

if __name__ == '__main__':
    main()
