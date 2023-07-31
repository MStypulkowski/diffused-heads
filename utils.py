import os
import tempfile
import scipy.io.wavfile as wav
import ffmpeg
import cv2
from PIL import Image

import decord
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, GaussianBlur, Grayscale, Resize
import torchaudio

decord.bridge.set_bridge('torch')
torchaudio.set_audio_backend("sox_io")


class AudioEncoder(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = torch.jit.load(path)
        self.register_buffer('hidden', torch.zeros(2, 1, 256))

    def forward(self, audio):
        self.reset()
        x = create_windowed_sequence(audio, 3200, cutting_stride=640, pad_samples=3200-640, cut_dim=1)
        embs = []                                           
        for i in range(x.shape[1]):
            emb, _, self.hidden = self.model(x[:, i], torch.LongTensor([3200]), init_state=self.hidden)
            embs.append(emb)
        return torch.vstack(embs)

    def reset(self):
        self.hidden = torch.zeros(2, 1, 256).to(self.hidden.device)


def get_audio_emb(audio_path, checkpoint, device):
    audio, audio_rate = torchaudio.load(audio_path, channels_first=False)
    assert audio_rate == 16000, 'Only 16 kHZ audio is supported.'
    audio = audio[None, None, :, 0].to(device)

    audio_encoder = AudioEncoder(checkpoint).to(device)

    emb = audio_encoder(audio)
    return audio, emb


def get_id_frame(path, random=False, resize=128):
    if path.endswith('.mp4'):
        vr = decord.VideoReader(path)
        if random:
            idx = [np.random.randint(len(vr))]
        else:
            idx = [0]
        frame = vr.get_batch(idx).permute(0, 3, 1, 2)
    else:
        frame = load_image_to_torch(path).unsqueeze(0)
    
    frame = (frame / 255) * 2 - 1
    frame = Resize((resize, resize), antialias=True)(frame).float()
    return frame


def get_motion_transforms(args):
    motion_transforms = []
    if args.motion_blur:
        motion_transforms.append(GaussianBlur(5, sigma=2.0))
    if args.grayscale_motion:
        motion_transforms.append(Grayscale(1))
    return Compose(motion_transforms)


def save_audio(path, audio, audio_rate=16000):
    if torch.is_tensor(audio):
        aud = audio.squeeze().detach().cpu().numpy()
    else:
        aud = audio.copy()  # Make a copy so that we don't alter the object

    aud = ((2 ** 15) * aud).astype(np.int16)
    wav.write(path, audio_rate, aud)


def save_video(path, video, fps=25, scale=2, audio=None, audio_rate=16000, overlay_pts=None, ffmpeg_experimental=False):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
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


def pad_both_ends(tensor, left, right, dim=0):
    no_dims = len(tensor.size())
    if dim == -1:
        dim = no_dims - 1

    padding = [0] * 2 * no_dims
    padding[2 * (no_dims - dim - 1)] = left
    padding[2 * (no_dims - dim - 1) + 1] = right
    return F.pad(tensor, padding, "constant", 0)


def cut_n_stack(seq, snip_length, cut_dim=0, cutting_stride=None, pad_samples=0):
    if cutting_stride is None:
        cutting_stride = snip_length

    pad_left = pad_samples // 2
    pad_right = pad_samples - pad_samples // 2

    seq = pad_both_ends(seq, pad_left, pad_right, dim=cut_dim)

    stacked = seq.narrow(cut_dim, 0, snip_length).unsqueeze(0)
    iterations = (seq.size()[cut_dim] - snip_length) // cutting_stride + 1
    for i in range(1, iterations):
        stacked = torch.cat((stacked, seq.narrow(cut_dim, i * cutting_stride, snip_length).unsqueeze(0)))
    return stacked


def create_windowed_sequence(seqs, snip_length, cut_dim=0, cutting_stride=None, pad_samples=0):
    windowed_seqs = []
    for seq in seqs:
        windowed_seqs.append(cut_n_stack(seq, snip_length, cut_dim, cutting_stride, pad_samples).unsqueeze(0))

    return torch.cat(windowed_seqs)