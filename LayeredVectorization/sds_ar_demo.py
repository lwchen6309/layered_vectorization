from typing import Optional, Tuple
import math
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from torch.optim.sgd import SGD

T = torch.Tensor


def ceil_to_multiple(x: int, base: int) -> int:
    return int(math.ceil(x / base) * base)


def load_image_ar(image_path: str) -> np.ndarray:
    return np.array(Image.open(image_path).convert('RGB'))


def resize_pad_ar(image: np.ndarray, min_side: int = 512, multiple: int = 8) -> Tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    scale = max(min_side / h, min_side / w, 1.0)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = np.array(Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BICUBIC))

    pad_h = ceil_to_multiple(new_h, multiple)
    pad_w = ceil_to_multiple(new_w, multiple)
    pad_top = (pad_h - new_h) // 2
    pad_bottom = pad_h - new_h - pad_top
    pad_left = (pad_w - new_w) // 2
    pad_right = pad_w - new_w - pad_left

    padded = np.pad(resized, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')
    meta = {
        'orig_h': h, 'orig_w': w,
        'resized_h': new_h, 'resized_w': new_w,
        'pad_top': pad_top, 'pad_bottom': pad_bottom,
        'pad_left': pad_left, 'pad_right': pad_right,
        'proc_h': pad_h, 'proc_w': pad_w,
    }
    return padded, meta


def unpad_and_restore(image: np.ndarray, meta: dict) -> np.ndarray:
    pt, pb = meta['pad_top'], meta['pad_bottom']
    pl, pr = meta['pad_left'], meta['pad_right']
    cropped = image[pt:image.shape[0]-pb, pl:image.shape[1]-pr]
    restored = np.array(Image.fromarray(cropped).resize((meta['orig_w'], meta['orig_h']), Image.Resampling.BICUBIC))
    return restored


@torch.no_grad()
def get_text_embeddings(device, pipe: StableDiffusionPipeline, text: str) -> T:
    tokens = pipe.tokenizer([text], padding='max_length', max_length=77, truncation=True,
                            return_tensors='pt', return_overflowing_tokens=True).input_ids.to(device)
    return pipe.text_encoder(tokens).last_hidden_state.detach()


@torch.no_grad()
def denormalize(image: T) -> np.ndarray:
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image[0]


@torch.no_grad()
def decode(latent: T, pipe: StableDiffusionPipeline) -> np.ndarray:
    image = pipe.vae.decode((1 / 0.18215) * latent, return_dict=False)[0]
    return denormalize(image)


def init_pipe(device, dtype, unet, scheduler) -> Tuple[UNet2DConditionModel, T, T]:
    with torch.inference_mode():
        alphas = torch.sqrt(scheduler.alphas_cumprod).to(device, dtype=dtype)
        sigmas = torch.sqrt(1 - scheduler.alphas_cumprod).to(device, dtype=dtype)
    for p in unet.parameters():
        p.requires_grad = False
    return unet, alphas, sigmas


class SDSLoss:
    def __init__(self, device, pipe: StableDiffusionPipeline, dtype=torch.float32):
        self.t_min = 50
        self.t_max = 950
        self.alpha_exp = 0
        self.sigma_exp = 0
        self.dtype = dtype
        self.unet, self.alphas, self.sigmas = init_pipe(device, dtype, pipe.unet, pipe.scheduler)
        self.prediction_type = pipe.scheduler.config.prediction_type

    def noise_input(self, z, eps=None, timestep: Optional[int] = None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(low=self.t_min, high=min(self.t_max, 1000) - 1,
                                     size=(b,), device=z.device, dtype=torch.long)
        if eps is None:
            eps = torch.randn_like(z)
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t, eps, timestep, alpha_t, sigma_t

    def get_eps_prediction(self, z_t: T, timestep: T, text_embeddings: T, alpha_t: T, sigma_t: T, guidance_scale=1.0):
        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            e_t = self.unet(latent_input, timestep, embedd).sample
            if self.prediction_type == 'v_prediction':
                e_t = torch.cat([alpha_t] * 2) * e_t + torch.cat([sigma_t] * 2) * latent_input
            e_t_uncond, e_t = e_t.chunk(2)
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
        pred_z0 = (z_t - sigma_t * e_t) / alpha_t
        return e_t, pred_z0

    def get_sds_loss(self, z: T, text_embeddings: T, timestep: Optional[int] = None, guidance_scale=0.0):
        with torch.inference_mode():
            z_t, eps, timestep, alpha_t, sigma_t = self.noise_input(z, timestep=timestep)
            e_t, _ = self.get_eps_prediction(z_t, timestep, text_embeddings, alpha_t, sigma_t,
                                             guidance_scale=guidance_scale)
            grad_z = (alpha_t ** self.alpha_exp) * (sigma_t ** self.sigma_exp) * (e_t - eps)
            grad_z = torch.nan_to_num(grad_z.detach(), 0.0, 0.0, 0.0)
            log_loss = (grad_z ** 2).mean()
        sds_loss = grad_z.clone() * z
        return sds_loss.sum() / (z.shape[2] * z.shape[3]), log_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--prompt', type=str, default=' ')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--outdir', type=str, default='sds_ar_demo_outputs')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        raise RuntimeError('This demo is intended for GPU/CUDA.')

    os = __import__('os')
    os.makedirs(args.outdir, exist_ok=True)

    image = load_image_ar(args.image)
    proc, meta = resize_pad_ar(image, min_side=512, multiple=8)

    pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5').to(device)
    sds_loss = SDSLoss(device, pipe)

    image_source = torch.from_numpy(proc).float().permute(2, 0, 1) / 127.5 - 1
    image_source = image_source.unsqueeze(0).to(device)

    with torch.no_grad():
        z_source = pipe.vae.encode(image_source)['latent_dist'].mean * 0.18215
        embedding_null = get_text_embeddings(device, pipe, '')
        embedding_text = get_text_embeddings(device, pipe, args.prompt)
        embedding_target = torch.stack([embedding_null, embedding_text], dim=1)

    z_target = z_source.clone().requires_grad_(True)
    optimizer = SGD(params=[z_target], lr=1e-1)

    Image.fromarray(image).save(f"{args.outdir}/orig.png")
    Image.fromarray(proc).save(f"{args.outdir}/padded_input.png")

    for i in range(args.steps):
        loss, log_loss = sds_loss.get_sds_loss(z_target, embedding_target)
        optimizer.zero_grad()
        (2000 * loss).backward()
        optimizer.step()

        out = decode(z_target, pipe)
        restored = unpad_and_restore(out, meta)
        Image.fromarray(restored).save(f"{args.outdir}/{i+1:03d}.png")
        print(f"step {i+1} loss={float(loss.item()):.6f} log_loss={float(log_loss.item()):.6f} proc=({meta['proc_h']},{meta['proc_w']}) orig=({meta['orig_h']},{meta['orig_w']})", flush=True)

    print('done', args.outdir)


if __name__ == '__main__':
    main()
