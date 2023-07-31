import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
from torchvision.transforms import Compose


class Diffusion(nn.Module):
    def __init__(
        self, nn_backbone, device, n_timesteps=1000, in_channels=3, image_size=128, out_channels=6, motion_transforms=None):
        super(Diffusion, self).__init__()

        self.nn_backbone = nn_backbone
        self.n_timesteps = n_timesteps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.x_shape = (image_size, image_size)
        self.device = device

        self.motion_transforms = motion_transforms if motion_transforms else Compose([])

        self.timesteps = torch.arange(n_timesteps)
        self.beta = self.get_beta_schedule()
        self.set_params()
        self.device = device

    def sample(self, x_cond, audio_emb, n_audio_motion_embs=2, n_motion_frames=2, motion_channels=3):
        with torch.no_grad():
            n_frames = audio_emb.shape[1]

            xT = torch.randn(x_cond.shape[0], n_frames, self.in_channels, self.x_shape[0], self.x_shape[1]).to(x_cond.device)

            audio_ids = [0] * n_audio_motion_embs
            for i in range(n_audio_motion_embs + 1):
                audio_ids += [i]
            
            motion_frames = [self.motion_transforms(x_cond) for _ in range(n_motion_frames)]
            motion_frames = torch.cat(motion_frames, dim=1)

            samples = []
            for i in trange(n_frames, desc=f'Sampling'):
                sample_frame = self.sample_loop(xT[:, i].to(x_cond.device), x_cond, motion_frames, audio_emb[:, audio_ids])
                samples.append(sample_frame.unsqueeze(1))
                motion_frames = torch.cat([motion_frames[:, motion_channels:, :], self.motion_transforms(sample_frame)], dim=1)
                audio_ids = audio_ids[1:] + [min(i + n_audio_motion_embs + 1, n_frames - 1)]
        return torch.cat(samples, dim=1)

    def sample_loop(self, xT, x_cond, motion_frames, audio_emb):
        xt = xT
        for i, t in reversed(list(enumerate(self.timesteps))):
            timesteps = torch.tensor([t] * xT.shape[0]).to(xT.device)
            timesteps_ids = torch.tensor([i] * xT.shape[0]).to(xT.device)
            nn_out = self.nn_backbone(xt, timesteps, x_cond, motion_frames=motion_frames, audio_emb=audio_emb)
            mean, logvar = self.get_p_params(xt, timesteps_ids, nn_out)
            noise = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
            xt = mean + noise * torch.exp(logvar / 2)

        return xt

    def get_p_params(self, xt, timesteps, nn_out):
        if self.in_channels == self.out_channels:
            eps_pred = nn_out
            p_logvar = self.expand(torch.log(self.beta[timesteps]))
        else:
            eps_pred, nu = nn_out.chunk(2, 1)
            nu = (nu + 1) / 2
            p_logvar = nu * self.expand(torch.log(self.beta[timesteps])) + (1 - nu) * self.expand(self.log_beta_tilde_clipped[timesteps])

        p_mean, _ = self.get_q_params(xt, timesteps, eps_pred=eps_pred)
        return p_mean, p_logvar

    def get_q_params(self, xt, timesteps, eps_pred=None, x0=None):
        if x0 is None:
        # predict x0 from xt and eps_pred
            coef1_x0 = self.expand(self.coef1_x0[timesteps])
            coef2_x0 = self.expand(self.coef2_x0[timesteps])
            x0 = coef1_x0 * xt - coef2_x0 * eps_pred
            x0 = x0.clamp(-1, 1)

        # q(x_{t-1} | x_t, x_0)
        coef1_q = self.expand(self.coef1_q[timesteps])
        coef2_q = self.expand(self.coef2_q[timesteps])
        q_mean = coef1_q * x0 + coef2_q * xt

        q_logvar = self.expand(self.log_beta_tilde_clipped[timesteps])

        return q_mean, q_logvar

    def get_beta_schedule(self, max_beta=0.999):
        alpha_bar = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        betas = []
        for i in range(self.n_timesteps):
            t1 = i / self.n_timesteps
            t2 = (i + 1) / self.n_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas).float()

    def set_params(self):
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_prev = torch.cat([torch.ones(1,), self.alpha_bar[:-1]])

        self.beta_tilde = self.beta * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.log_beta_tilde_clipped = torch.log(torch.cat([self.beta_tilde[1, None], self.beta_tilde[1:]]))

        # to caluclate x0 from eps_pred
        self.coef1_x0 = torch.sqrt(1.0 / self.alpha_bar)
        self.coef2_x0 = torch.sqrt(1.0 / self.alpha_bar - 1)

        # for q(x_{t-1} | x_t, x_0)
        self.coef1_q = self.beta * torch.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.coef2_q = (1.0 - self.alpha_bar_prev) * torch.sqrt(self.alpha) / (1.0 - self.alpha_bar)

    def space(self, n_timesteps_new):
        # change parameters for spaced timesteps during sampling
        self.timesteps = self.space_timesteps(self.n_timesteps, n_timesteps_new)
        self.n_timesteps = n_timesteps_new

        self.beta = self.get_spaced_beta()
        self.set_params()

    def space_timesteps(self, n_timesteps, target_timesteps):
        all_steps = []
        frac_stride = (n_timesteps - 1) / (target_timesteps - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(target_timesteps):
            taken_steps.append(round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        return all_steps

    def get_spaced_beta(self):
        last_alpha_cumprod = 1.0
        new_beta = []
        for i, alpha_cumprod in enumerate(self.alpha_bar):
            if i in self.timesteps:
                new_beta.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        return torch.tensor(new_beta)

    def expand(self, arr, dim=4):
        while arr.dim() < dim:
            arr = arr[:, None]
        return arr.to(self.device)