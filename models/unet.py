"""
Code is a simplified version of https://github.com/openai/guided-diffusion
"""
import torch
import torch.nn as nn

from models.blocks import (
    zero_module, CondResBlock, ResBlock, CondSequential, 
    AttentionBlock, CondAttentionBlock, TimestepEmbedding, Upsample, Downsample
)


class UNet(nn.Module):
    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions,
                dropout=0, channel_mult=(1, 2, 4, 8), num_heads=1, num_head_channels=-1, resblock_updown=False, 
                id_condition_type='frame', audio_condition_type='pre_gn', precision=32, n_motion_frames=0, grayscale_motion=False,
                n_audio_motion_embs=0):
        super(UNet, self).__init__()

        self.image_size = image_size
        self.id_condition_type = id_condition_type
        self.audio_condition_type = audio_condition_type
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = [image_size // res for res in attention_resolutions]
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.dtype = torch.float32 if precision == 32 else torch.float16
        self.n_motion_frames = n_motion_frames
        self.n_audio_motion_embs = n_audio_motion_embs
        self.img_channels = in_channels

        self.motion_channels = 1 if grayscale_motion else 3
        self.in_channels = in_channels + self.motion_channels * n_motion_frames
        if id_condition_type == 'frame':
            self.in_channels += in_channels
        
        time_embed_dim = model_channels * 4
        self.identity_encoder = None
        if id_condition_type != 'frame':
            self.identity_encoder = IdentityEncoder(
                image_size, in_channels, model_channels, time_embed_dim, num_res_blocks, self.attention_resolutions, 
                dropout=dropout, channel_mult=channel_mult, num_heads=num_heads, num_head_channels=num_head_channels,
                resblock_updown=resblock_updown, precision=precision
            )
        
        self.time_embed = TimestepEmbedding(model_channels)
        
        self.audio_embed = nn.Sequential(
            nn.Linear(model_channels * (2 * n_audio_motion_embs + 1), 4 * model_channels),
            nn.SiLU(),
            nn.Linear(4 * model_channels, 4 * model_channels),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [CondSequential(nn.Conv2d(self.in_channels, ch, 3, padding=1))]
        )
        
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [CondResBlock(ch, time_embed_dim, dropout, out_channels=int(mult * model_channels), id_condition_type=id_condition_type, audio_condition_type=audio_condition_type)]

                ch = int(mult * model_channels)
                if ds in self.attention_resolutions:
                    if audio_condition_type == 'attention':
                        layers.append(CondAttentionBlock(ch, 4 * model_channels, image_size // ds, num_heads=num_heads, num_head_channels=num_head_channels))
                    else:
                        layers.append(AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels))

                self.input_blocks.append(CondSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    CondSequential(CondResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, down=True, id_condition_type=id_condition_type, audio_condition_type=audio_condition_type))
                        if resblock_updown
                        else Downsample()
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        
        if audio_condition_type == 'attention':
            attention_layer = CondAttentionBlock(ch, 4 * model_channels, image_size // ds, num_heads=num_heads, num_head_channels=num_head_channels)
        else:
            attention_layer = AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels)
        self.middle_block = CondSequential(
            CondResBlock(ch, time_embed_dim, dropout, id_condition_type=id_condition_type, audio_condition_type=audio_condition_type),
            attention_layer,
            CondResBlock(ch, time_embed_dim, dropout, id_condition_type=id_condition_type, audio_condition_type=audio_condition_type)
        )

        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    CondResBlock(ch + ich, time_embed_dim, dropout, out_channels=int(model_channels * mult), id_condition_type=id_condition_type, audio_condition_type=audio_condition_type)
                ]
                ch = int(model_channels * mult)
                if ds in self.attention_resolutions:
                    if audio_condition_type == 'attention':
                        layers.append(CondAttentionBlock(ch, 4 * model_channels, image_size // ds, num_heads=num_heads, num_head_channels=num_head_channels))
                    else:
                        layers.append(AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        CondResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, up=True, id_condition_type=id_condition_type, audio_condition_type=audio_condition_type)
                        if resblock_updown
                        else Upsample()
                    )
                    ds //= 2
                self.output_blocks.append(CondSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, x_cond, motion_frames=None, audio_emb=None):
        t_emb = self.time_embed(timesteps, dtype=x.dtype)

        if audio_emb is not None:
            a_emb = audio_emb.reshape(audio_emb.shape[0], -1)
            a_emb = self.audio_embed(a_emb)
        else:
            a_emb = 0

        if motion_frames is not None:
            x = torch.cat([x, motion_frames], dim=1)

        if self.id_condition_type == 'frame':
            x = torch.cat([x, x_cond], dim=1)
            if self.audio_condition_type == 'pre_gn':
                emb = t_emb + a_emb
            elif self.audio_condition_type in ['post_gn', 'double_pre_gn', 'attention'] and audio_emb is not None:
                emb = (t_emb, a_emb)
            else:
                raise NotImplemented(self.audio_condition_type)

        elif self.id_condition_type == 'post_gn':
            x_emb = self.identity_encoder(x_cond)
            if self.audio_condition_type == 'pre_gn':
                emb = (t_emb + a_emb, x_emb)
            elif self.audio_condition_type in ['post_gn', 'double_pre_gn', 'attention'] and audio_emb is not None:
                emb = (t_emb, a_emb, x_emb)
            else:
                raise NotImplementedError(self.audio_condition_type)

        elif self.id_condition_type == 'pre_gn':
            x_emb = self.identity_encoder(x_cond)
            if self.audio_condition_type == 'pre_gn':
                emb = x_emb + t_emb + a_emb
            elif self.audio_condition_type in ['post_gn', 'double_pre_gn', 'attention'] and audio_emb is not None:
                emb = (x_emb + t_emb, a_emb)
            else:
                raise NotImplementedError(self.audio_condition_type)
                
        else:
            raise NotImplementedError(self.id_condition_type)

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        return self.out(h)


class IdentityEncoder(nn.Module):
    def __init__(self, image_size, in_channels, model_channels, out_dim, num_res_blocks, attention_resolutions,
                dropout=0, channel_mult=(1, 2, 4, 8), num_heads=1, num_head_channels=-1, resblock_updown=False, 
                precision=32):
        super(IdentityEncoder, self).__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_dim = out_dim
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.dtype = torch.float32 if precision == 32 else torch.float16

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(self.in_channels, ch, 3, padding=1))]
        )

        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, dropout, out_channels=int(mult * model_channels))]

                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels))

                self.input_blocks.append(nn.Sequential(*layers))

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(ResBlock(ch, dropout, out_channels=out_ch, down=True))
                        if resblock_updown
                        else Downsample()
                )
                ch = out_ch
                ds *= 2

        self.middle_block = nn.Sequential(
            ResBlock(ch, dropout),
            AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels),
            ResBlock(ch, dropout)
        )

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            zero_module(nn.Conv2d(ch, out_dim, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        for module in self.input_blocks:
            x = module(x)

        x = self.middle_block(x)

        return self.out(x)


if __name__ == '__main__':

    image_size = 64
    in_channels = 3
    model_channels = 64
    out_channels = 6 # 3 or 6 if sigma learnable
    num_res_blocks = 1
    attention_resolutions = (8, 4, 2)
    dropout = 0.1
    channel_mult = (1, 2, 3)
    num_heads = 4
    num_head_channels = -1
    resblock_updown = True

    unet = UNet(image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions,
                dropout=dropout, channel_mult=channel_mult, num_heads=num_heads, num_head_channels=-1, resblock_updown=True, 
                id_condition_type='frame', precision=32).to('cuda')
    print(unet)
    x = torch.randn(5, 3, 64, 64).to('cuda')
    t = torch.randint(10, (5,)).to('cuda')
    out = unet(x, t)
    print(out.shape)
