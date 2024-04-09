import os
from io import BytesIO
import tempfile
from datetime import datetime
import scipy.io.wavfile as wav
import ffmpeg
import cv2
from PIL import Image

import numpy as np
import torch
from torchvision.transforms import Compose, GaussianBlur, Grayscale
from pytorch_lightning.callbacks import ModelCheckpoint


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(CustomModelCheckpoint, self).__init__(*args, **kwargs)

    def _save_checkpoint(self, *args, **kwargs):
        tries = 5
        while tries > 0:
            try:
                super(CustomModelCheckpoint, self)._save_checkpoint(*args, **kwargs)
                break
            except:
                tries -= 1


def get_exp_name(args, mode='train'):
    if mode == 'test':
        epoch = args.checkpoint.split('/')[-1].split('-')[0]
        exp_name = args.checkpoint.split('/')[-4]
        return 'Test ' + epoch + ' ' + exp_name
    name = datetime.now().strftime("%Y-%m-%d %H:%M")
    name += ' ' + args.data_dir.split('/')[-1]
    name += ' res_' + str(args.img_resize)
    name += ' n_audio_embs_' + str(args.n_audio_motion_embs)
    name += ' grayscale' if args.grayscale_motion else ''
    name += ' att_res_[' + ','.join([str(att_res) for att_res in args.attention_resolutions]) + ']'
    name += ' audio_cond_' + args.audio_condition_type
    name += ' vlb_weight_' + str(args.vlb_weight)
    name += ' lip_weight_' + str(args.lip_weight)
    name += ' bsz_' + str(args.bsz)
    name += ' lr_' + str(args.lr)
    return name


def pad(seq, max_len):
    padding = torch.zeros(max_len - seq.shape[0], seq.shape[1])
    return torch.cat([seq, padding], dim=0)


def pad_sequences(seq_list, max_len):
    # list of sequnces of shape (n_i, d)
    padded_seq_list = []
    for seq in seq_list:
        if seq.shape[0] < max_len:
            padded_seq = pad(seq, max_len)
            padded_seq_list.append(padded_seq.unsqueeze(0))
        elif seq.shape[0] == max_len:
            padded_seq_list.append(seq.unsqueeze(0))
        else:
            raise ValueError
    
    return torch.cat(padded_seq_list, dim=0)


def get_motion_transforms(args):
    motion_transforms = []
    if args.motion_blur:
        motion_transforms.append(GaussianBlur(5, sigma=2.0))
    if args.grayscale_motion:
        motion_transforms.append(Grayscale(1))
    return Compose(motion_transforms)


def count_trainable_parameters(model):
    count = 0
    for p in model.parameters():
        if p.requires_grad:
            i = 0
            local_count = 1
            while i < p.dim():
                local_count *= p.shape[i]
                i += 1
            count += local_count
    
    return count


def save_img(path, img):
    if torch.is_tensor(img):
        img = img.squeeze().detach().cpu().numpy()
    img = np.rollaxis(255 * (img * 0.5 + 0.5), 0, 3).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)


def save_audio(path, audio, audio_rate=16000):
    if torch.is_tensor(audio):
        aud = audio.squeeze().detach().cpu().numpy()
    else:
        aud = audio.copy()  # Make a copy so that we don't alter the object

    aud = ((2 ** 15) * aud).astype(np.int16)
    wav.write(path, audio_rate, aud)


def save_video(path, video, fps=25, scale=2, audio=None, audio_rate=16000, overlay_pts=None, ffmpeg_experimental=False):
    success = True    
    out_size = (scale * video.shape[-1], scale * video.shape[-2])
    video_path = get_temp_path(os.path.split(path)[0], ext=".mp4")
    if torch.is_tensor(video):
        vid = video.squeeze().detach().cpu().numpy()
    else:
        vid = video.copy()  # Make a copy so that we don't alter the object

    if np.min(vid) < 0:
        vid = 127 * vid + 127
    elif np.max(vid) <= 1:
        vid = 255 * vid

    is_color = True
    if vid.ndim == 3:
        is_color = False

    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), out_size, isColor=is_color)
    for i, frame in enumerate(vid):
        if is_color:
            frame = cv2.cvtColor(np.rollaxis(frame, 0, 3), cv2.COLOR_RGB2BGR)

        if scale != 1:
            frame = cv2.resize(frame, out_size)

        write_frame = frame.astype('uint8')

        if overlay_pts is not None:
            for pt in overlay_pts[i]:
                cv2.circle(write_frame, (int(scale * pt[0]), int(scale * pt[1])), 2, (0, 0, 0), -1)

        writer.write(write_frame)
    writer.release()

    inputs = [ffmpeg.input(video_path)['v']]

    if audio is not None:  # Save the audio file
        audio_path = swp_extension(video_path, ".wav")
        save_audio(audio_path, audio, audio_rate)
        inputs += [ffmpeg.input(audio_path)['a']]

    try:
        if ffmpeg_experimental:
            out = ffmpeg.output(*inputs, path, strict='-2', loglevel="panic", vcodec='h264').overwrite_output()
        else:
            out = ffmpeg.output(*inputs, path, loglevel="panic", vcodec='h264').overwrite_output()
        out.run(quiet=True)
    except:
        success = False

    if audio is not None and os.path.isfile(audio_path):
        os.remove(audio_path)
    if os.path.isfile(video_path):
        os.remove(video_path)

    return success


def video_to_stream(temp_file, keep_files=False):
    stream = BytesIO(open(temp_file, "rb").read())

    if not keep_files and os.path.isfile(temp_file):
        os.remove(temp_file)

    return stream


def load_img(temp_file):
    img = Image.open(temp_file)

    if os.path.isfile(temp_file):
        os.remove(temp_file)
    
    return img


def load_image_to_torch(dir):
    img = Image.open(dir).convert('RGB')
    img = np.array(img)
    return torch.from_numpy(img).permute(2, 0, 1)


def get_temp_path(tmp_dir, mode="", ext=""):
    file_path = next(tempfile._get_candidate_names()) + mode + ext
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    file_path = os.path.join(tmp_dir, file_path)
    return file_path


def swp_extension(file, ext):
    return os.path.splitext(file)[0] + ext