import hydra
import torch

from diffusion import Diffusion
from utils import get_id_frame, get_audio_emb, save_video


@hydra.main(config_path='.', config_name='config_crema', version_base='1.1')
def main(args):
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    print('Loading model...')
    unet = torch.jit.load(args.checkpoint)
    diffusion = Diffusion(unet, device, **args.diffusion).to(device)
    diffusion.space(args.inference_steps)

    id_frame = get_id_frame(args.id_frame, random=args.id_frame_random, resize=args.diffusion.image_size).to(device)
    audio, audio_emb = get_audio_emb(args.audio, args.encoder_checkpoint, device)

    samples = diffusion.sample(id_frame, audio_emb.unsqueeze(0), **args.unet)

    save_video(args.output, samples, audio=audio, fps=25, audio_rate=16000)
    print(f'Results saved at {args.output}')


if __name__ == '__main__':
    main()