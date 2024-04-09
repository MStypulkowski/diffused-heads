import os
import random
from math import ceil
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import decord

decord.bridge.set_bridge('torch')
torchaudio.set_audio_backend("sox_io")


class AudioVisualDataset(Dataset):
    def __init__(self, data_dir, split=None, identity_frame='first', frame_transforms=None, 
            motion_transforms=None, audio_emb_dir=None, n_motion_frames=0, n_audio_motion_embs=0, 
            img_resize=None, landmarks=False, effective_bsz=None, 
            check_for_existing_samples=False, log_dir=None, frame_limit=None):
        self.data_dir = data_dir
        self.identity_frame = identity_frame
        self.frame_transforms = frame_transforms
        self.motion_transforms = motion_transforms
        self.audio_emb_dir = audio_emb_dir
        self.split = split
        self.n_motion_frames = n_motion_frames
        self.n_audio_motion_embs = n_audio_motion_embs
        self.img_resize = img_resize
        self.landmarks = landmarks
        self.frame_limit = frame_limit

        if split == 'val':
            split_file = 'train'
        else:
            split_file = split

        if data_dir.split('/')[-1] == 'vox':
            self.video_dir = os.path.join(data_dir, 'videos', 'train')
            self.audio_dir = os.path.join(data_dir, 'audio', 'train')
            self.lmks_dir = os.path.join(data_dir, 'lmks', 'train')
            self.file_list_dir = os.path.join(data_dir, f'file_list_{split_file}_new.txt')
            self.audio_ext = '.mp3'
        elif data_dir.split('/')[-1] == 'avspeech':
            self.video_dir = os.path.join(data_dir, 'cropped_videos')
            self.audio_dir = os.path.join(data_dir, 'audio')
            self.lmks_dir = os.path.join(data_dir, 'low_acc_smooth_3d_landmarks')
            self.file_list_dir = os.path.join(data_dir, f'file_list_{split_file}_new.txt')
            self.audio_ext = '.wav'
        else:
            self.video_dir = os.path.join(data_dir, 'video')
            self.audio_dir = os.path.join(data_dir, 'audio')
            self.lmks_dir = os.path.join(data_dir, 'lmks')
            self.file_list_dir = os.path.join(data_dir, f'file_list_{split_file}.txt')
            self.audio_ext = '.wav'
            
        with open(self.file_list_dir, 'r') as f:
            self.file_list = []
            for file_name in f:
                self.file_list.append(file_name[:-5])
        
        self.set_fps_ar()

        if check_for_existing_samples:
            print('Old file list', len(self.file_list))
            new_file_list = []
            log_dir = os.path.join(log_dir, self.split)
            for file_name in self.file_list:
                sample_dir = os.path.join(log_dir, file_name + '.mp4')
                if not os.path.exists(sample_dir):
                    new_file_list.append(file_name)
            self.file_list = new_file_list
            print('New file list', len(self.file_list))

        if effective_bsz is not None:
            # sample bsz * n_GPUs videos for val and test splits during evaluation step
            self.file_list = self.file_list[:effective_bsz]

    def __getitem__(self, idx):
        file_name = self.file_list[idx]

        video_file_dir = os.path.join(self.video_dir, file_name + '.mp4')
        tries = 5
        while tries > 0:
                try:
                    vr = decord.VideoReader(video_file_dir)
                    break
                except:
                    tries -= 1

        if self.audio_emb_dir:
            audio_emb_dir = os.path.join(self.audio_emb_dir, file_name + '.pth')
            tries = 5
            while tries > 0:
                try:
                    audio_emb = torch.load(audio_emb_dir)
                    break
                except:
                    tries -= 1
            max_frame = min(audio_emb.shape[0], len(vr))
        else:
            max_frame = len(vr)
        
        ids = self.get_ids(max_frame, identity_frame=self.identity_frame)
        frame_ids = ids['frames']
        audio_ids = ids['audio']

        frames = vr.get_batch(frame_ids).permute(0, 3, 1, 2)
        frames = (frames / 255) * 2 - 1
        original_res = frames.shape[2:]

        if self.frame_transforms:
            frames = self.frame_transforms(frames)

        if self.landmarks:
            landmark_dir = os.path.join(self.lmks_dir, file_name + '.npy')
            tries = 5
            while tries > 0:
                try:
                    landmarks = np.load(landmark_dir)[frame_ids[1], 48:68, :2] # take landmarks only for lips, without Z axis
                    break
                except:
                    tries -= 1
            ratio = torch.tensor(original_res) / self.img_resize
            landmarks /= ratio
        
        if self.audio_emb_dir:
            if self.split == 'train':
                motion_frames = None
                if self.n_motion_frames > 0:
                    if self.motion_transforms:
                        motion_frames = self.motion_transforms(frames[2:])
                    else:
                        motion_frames = frames[2:]
                    motion_frames = torch.cat(list(motion_frames), dim=0)
                if self.landmarks:
                    return frames[0], frames[1], motion_frames, audio_emb[audio_ids], landmarks
                return frames[0], frames[1], motion_frames, audio_emb[audio_ids]
                
            elif self.split == 'test' or self.split == 'val':
                audio_file_dir = os.path.join(self.audio_dir, file_name + self.audio_ext)
                audio, audio_rate = torchaudio.load(audio_file_dir, channels_first=False)
                if self.frame_limit is not None:
                    audio_emb = audio_emb[:self.frame_limit]
                    audio = audio[:self.frame_limit * self.audio_rate // self.video_rate]
                return frames[1], audio_emb, audio, file_name
        return frames[0], frames[1]

    def __len__(self):
        return len(self.file_list)

    def get_ids(self, n_frames, identity_frame=None):
        # returns GT frame, ID frame, maybe motion frames, and audio motion frames
        ids = {}
        if identity_frame == 'first':
            frame_ids= [np.random.randint(1, n_frames)] + [0]
        elif identity_frame == 'random':
            frame_ids = list(np.random.choice(n_frames, 2, replace=False))
        
        if self.n_motion_frames > 0:
            # get motion frames from the past only
            frame_ids += [frame_ids[1]] * (max(self.n_motion_frames - frame_ids[0], 0))
            for i in range(max(frame_ids[0] - self.n_motion_frames, 0), frame_ids[0]):
                frame_ids += [i]
        ids['frames'] = frame_ids

        audio_ids = []
        # get audio embs from both sides of the GT frame
        audio_ids += [0] * max(self.n_audio_motion_embs - frame_ids[0], 0)
        for i in range(max(frame_ids[0] - self.n_audio_motion_embs, 0), min(frame_ids[0] + self.n_audio_motion_embs + 1, n_frames)):
            audio_ids += [i]
        audio_ids += [n_frames - 1] * max(frame_ids[0] + self.n_audio_motion_embs - n_frames + 1, 0)
        ids['audio'] = audio_ids

        return ids

    def set_fps_ar(self):
        # set fps and audio rate
        video_file_dir = os.path.join(self.video_dir, self.file_list[0] + '.mp4')
        vr = decord.VideoReader(video_file_dir)
        self.video_rate = ceil(vr.get_avg_fps())

        audio_file_dir = os.path.join(self.audio_dir, self.file_list[0] + self.audio_ext)
        _, self.audio_rate = torchaudio.load(audio_file_dir, channels_first=False)