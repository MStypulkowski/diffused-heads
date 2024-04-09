import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, GaussianBlur, Grayscale
import pytorch_lightning as pl

from datasets import AudioVisualDataset
from utils import pad_sequences


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, bsz, img_resize, 
            n_workers=0, identity_frame=None, audio_emb_dir=None,
            n_motion_frames=0, motion_transforms=None,
            n_audio_motion_embs=0, landmarks=False, effective_bsz=None, 
            check_for_existing_samples=False, log_dir=None, frame_limit=None):
        super(DataModule, self).__init__()

        self.data_dir = data_dir
        self.bsz=bsz
        self.n_workers = n_workers
        self.identity_frame = identity_frame

        frame_transforms = Compose([Resize(img_resize)])

        self.dataset_train = AudioVisualDataset(
            self.data_dir, split='train',
            identity_frame=self.identity_frame, frame_transforms=frame_transforms,
            motion_transforms=motion_transforms, audio_emb_dir=audio_emb_dir, 
            n_motion_frames=n_motion_frames, n_audio_motion_embs=n_audio_motion_embs,
            img_resize=img_resize, landmarks=landmarks)

        self.dataset_val = AudioVisualDataset(
            self.data_dir, split='val',
            identity_frame=self.identity_frame, frame_transforms=frame_transforms, 
            audio_emb_dir=audio_emb_dir, effective_bsz=effective_bsz, frame_limit=frame_limit)

        self.dataset_test = AudioVisualDataset(
            self.data_dir, split='test',
            identity_frame=self.identity_frame, frame_transforms=frame_transforms, 
            audio_emb_dir=audio_emb_dir, effective_bsz=effective_bsz, 
            check_for_existing_samples=check_for_existing_samples, log_dir=log_dir, frame_limit=frame_limit)

        self.video_rate = self.dataset_train.video_rate
        self.audio_rate = self.dataset_train.audio_rate

    def train_dataloader(self):
        return DataLoader(self.dataset_train, num_workers=self.n_workers, batch_size=self.bsz, shuffle=True, drop_last=True)

    def val_dataloader(self):
        test_loader = DataLoader(self.dataset_test, num_workers=self.n_workers, batch_size=self.bsz, shuffle=True, drop_last=True, collate_fn=collate)
        val_loader = DataLoader(self.dataset_val, num_workers=self.n_workers, batch_size=self.bsz, shuffle=True, drop_last=True, collate_fn=collate)
        return [test_loader, val_loader]
    
    def test_dataloader(self):
        return DataLoader(self.dataset_test, num_workers=self.n_workers, batch_size=self.bsz, shuffle=True, drop_last=False, collate_fn=collate)


def collate(batch):
    id_frame_list = []
    audio_emb_list = []
    audio_list = []
    file_name_list = []

    for item in batch:
        id_frame_list.append(item[0])
        audio_emb_list.append(item[1])
        audio_list.append(item[2])
        file_name_list.append(item[3])
    
    seq_lens = [audio_emb.shape[0] for audio_emb in audio_emb_list]

    audio_embs_padded = pad_sequences(audio_emb_list, max(seq_lens))

    return torch.stack(id_frame_list, dim=0), audio_embs_padded, audio_list, seq_lens, file_name_list