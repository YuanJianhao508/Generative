#adapeted from OPENAI GLIDE inpaint
from typing import Tuple
import os
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import functional as TF

def return_images(batch):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    img = Image.fromarray(reshaped.numpy())
    return img
    # img.save(os.path.join(output_dir, save_name))

def read_image(path: str, size: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    pil_img = Image.open(path).convert('RGB')
    pil_img = pil_img.resize((size, size), resample=Image.BICUBIC)
    img = np.array(pil_img)
    return torch.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1

def threshold_mask(pil_mask,size,device,alpha=0.25):
  mask = TF.to_tensor(pil_mask.resize((size, size), Image.BILINEAR))
  mask = mask.unsqueeze(0)
  mask_dist = TF.to_tensor(pil_mask.resize((size, size), Image.BILINEAR)).to(device).unsqueeze(0)
  #Threshold on the average of the mask
  std, mean = torch.std_mean(mask_dist.view(-1)[torch.nonzero(mask_dist.view(-1))])
  std = std.item()
  mean = mean.item()
  mask = mask.lt(mean+alpha*std).float()
  mask = 1-mask
  return mask


def glide_intervention(img_path,mask,model,options,diffusion,model_up,options_up,diffusion_up,prompt,batch_size,guidance_scale,upsample_temp,device):
    # Create an classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def denoised_fn(x_start):
        # Force the model to have the exact right x_start predictions
        # for the part of the image which is known.
        return (
                x_start * (1 - model_kwargs['inpaint_mask'])
                + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask']
        )
    source_image_256 = read_image(img_path, size=256)
    source_image_64 = read_image(img_path, size=64)
    source_mask_256 = threshold_mask(mask,256,device)
    source_mask_64 = threshold_mask(mask,64,device)
    # Create the text tokens to feed to the model.
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(
        tokens, options['text_ctx']
    )

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options['text_ctx']
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=torch.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=torch.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=torch.bool,
            device=device,
        ),

        # Masked inpainting image
        inpaint_image=(source_image_64 * source_mask_64).repeat(full_batch_size, 1, 1, 1).to(device),
        inpaint_mask=source_mask_64.repeat(full_batch_size, 1, 1, 1).to(device),
    )

    # Sample from the base model.
    model.del_cache()
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=False,
        model_kwargs=model_kwargs,
        cond_fn=None,
        denoised_fn=denoised_fn,
    )[:batch_size]
    model.del_cache()

    ##############################
    # Upsample the 64x64 samples #
    ##############################

    tokens = model_up.tokenizer.encode(prompt)
    tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
        tokens, options_up['text_ctx']
    )

    # Create the model conditioning dict.
    model_kwargs = dict(
        # Low-res image to upsample.
        low_res=((samples + 1) * 127.5).round() / 127.5 - 1,

        # Text tokens
        tokens=torch.tensor(
            [tokens] * batch_size, device=device
        ),
        mask=torch.tensor(
            [mask] * batch_size,
            dtype=torch.bool,
            device=device,
        ),

        # Masked inpainting image.
        inpaint_image=(source_image_256 * source_mask_256).repeat(batch_size, 1, 1, 1).to(device),
        inpaint_mask=source_mask_256.repeat(batch_size, 1, 1, 1).to(device),
    )
    # Sample from the base model.
    model_up.del_cache()
    up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
    up_samples = diffusion_up.p_sample_loop(
        model_up,
        up_shape,
        noise=torch.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=False,
        model_kwargs=model_kwargs,
        cond_fn=None,
        denoised_fn=denoised_fn,
    )[:batch_size]
    model_up.del_cache()


    return return_images(up_samples)