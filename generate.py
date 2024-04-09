import os
import yaml
import hydra
from datetime import datetime
from tqdm import tqdm
import wandb
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from lightning_modules import LightningDiffusion, DataModule
from utils import get_exp_name


@hydra.main(config_path='./configs', config_name='config_gen_test', version_base='1.1')
def main(args):
    # wandb.login()
    # wandb_logger = WandbLogger(
    #     project='', entity='', 
    #     name=get_exp_name(args, mode='test'), 
    #     settings=wandb.Settings(start_method="fork")
    #     )
    wandb_logger = None

    print(args)

    pl.seed_everything(2137)

    model = LightningDiffusion.load_from_checkpoint(args.checkpoint)
    model.diffusion.space(args.n_sample_timesteps)

    epoch = args.checkpoint.split('/')[-1].split('-')[0].split('=')[-1]
    args.log_dir = os.path.join(args.log_dir, 'test_videos', epoch)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    model.hparams.log_dir = args.log_dir

    if args.n_batches == -1:
        effective_bsz = None
    else:
        effective_bsz = args.n_batches * args.bsz * args.n_nodes * torch.cuda.device_count()

    data_module = DataModule(
        args.data_dir, args.bsz, model.hparams.image_size, 
        n_workers=args.n_workers, identity_frame=args.identity_frame,
        audio_emb_dir=args.audio_emb_dir, n_motion_frames=model.hparams.n_motion_frames, 
        motion_transforms=model.hparams.motion_transforms, 
        n_audio_motion_embs=model.hparams.n_audio_motion_embs, 
        effective_bsz=effective_bsz, check_for_existing_samples=args.check_for_existing_samples,
        log_dir=args.log_dir, frame_limit=args.frame_limit
        )

    # test_dataloader = DataLoader(data_module.dataset_test)

    print('\n', model.hparams, '\n')

    callbacks = [TQDMProgressBar(refresh_rate=10)]

    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=-1,
        strategy=DDPStrategy(find_unused_parameters=False),
        num_nodes=args.n_nodes,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=20,
        num_sanity_val_steps=0
    )

    trainer.test(model, data_module)

    if args.log_to_wandb:
        model.log_media('test', tmp_dir=args.log_dir, keep_files=args.keep_files)

if __name__ == '__main__':
    main()