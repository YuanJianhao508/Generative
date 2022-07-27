import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import math
import io
import sys

# from IPython import display
from omegaconf import OmegaConf
from PIL import Image
import requests
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

import getMask
sys.path.append('./taming-transformers')

import clip
from taming.models import cond_transformer, vqgan
import kornia.augmentation as K


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


def spherical_dist(x, y, noise=False, noise_coeff=0.1):
    x_normed = F.normalize(x, dim=-1)
    y_normed = F.normalize(y, dim=-1)
    if noise:
        with torch.no_grad():
            noise1 = torch.empty(x_normed.shape).normal_(0, 0.0422).to(x_normed).detach() * noise_coeff
            noise2 = torch.empty(y_normed.shape).normal_(0, 0.0422).to(x_normed).detach() * noise_coeff

            x_normed += noise1
            y_normed += noise2
    x_normed = F.normalize(x_normed, dim=-1)
    y_normed = F.normalize(y_normed, dim=-1)

    return x_normed.sub(y_normed).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def bdot(a, b):
    B = a.shape[0]
    S = a.shape[1]
    b = b.expand(B, -1)
    # print(a.shape)
    # print(b.shape)
    return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)


def inner_dist(x, y):
    x_normed = F.normalize(x, dim=-1)
    y_normed = F.normalize(y, dim=-1)
    return bdot(x_normed, y_normed)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.noise_fac = 0.1
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2, p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            K.RandomGrayscale(p=0.1),
        )

    def set_cut_pow(self, cut_pow):
        self.cut_pow = cut_pow

    def forward(self, input, cut_pow=None, augs=True, grads=True):
        if cut_pow is None:
            cut_pow = self.cut_pow

        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = torch.cat(cutouts, dim=0)
        if grads:
            batch = clamp_with_grad(batch, 0, 1)
        if augs:
            batch = self.augs(batch)
            if self.noise_fac:
                facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
                batch = batch + facs * torch.randn_like(batch)
        return batch


def load_vqgan_model(config_path, checkpoint_path):
    global gumbel
    gumbel = False
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
        gumbel = True
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def size_to_fit(size, max_dim, scale_up=False):
    w, h = size
    if not scale_up and max(h, w) <= max_dim:
        return w, h
    new_w, new_h = max_dim, max_dim
    if h > w:
        new_w = round(max_dim * w / h)
    else:
        new_h = round(max_dim * h / w)
    return new_w, new_h


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

from_image_path = './land_land.jpg'  #@param {type:"string"}
image_size =   224#@param {type:"integer"}
subject = 'bird'
from_text = 'a photo of a bird in forest background'  #@param {type:"string"}
to_text = 'a photo of a bird in ocean background'  #@param {type:"string"}
scale_dir_by =   1#@param {type:"number"}
clip_model = 'ViT-B/16'  #@param ["ViT-B/32", "ViT-B/16", "RN50", "RN101", "RN50x4", "RN50x16"]
use_mask = True  #@param {type:"boolean"}
invert_mask = False  #@param {type:"boolean"}
cut_pow_start = 0.3 #@param {type:"number"}
cut_pow_end =  1.0#@param {type:"number"}
cut_pow_length =  400#@param {type:"integer"}
mask_samples =  1#@param {type:"integer"}

#Set up model and devices

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
torch.cuda.empty_cache()
# sdbo= scale_dir_by
t = 0
cut_out_num=64

model = load_vqgan_model('/datasets/jianhaoy/checkpoints/vqgan_imagenet_f16_16384.yaml', '/datasets/jianhaoy/checkpoints/vqgan_imagenet_f16_16384.ckpt')
model.to(device)

perceptor, preprocess = clip.load(clip_model, jit=False)
perceptor.eval().requires_grad_(False).to(device)

torch.manual_seed(0)

pil_image = Image.open(fetch(from_image_path)).convert('RGB')
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

cut_size = perceptor.visual.input_resolution
f = 2**(model.decoder.num_resolutions - 1)
make_cutouts = MakeCutouts(cut_size, cut_out_num, cut_pow=cut_pow_start)

pil_mask = getMask.get_mask(perceptor,preprocess,subject,from_image_path,device=device,mask_samples=mask_samples)

# Properly rescale image
sideX, sideY = size_to_fit(pil_image.size, image_size, True)
toksX, toksY = sideX // f, sideY // f
sideX, sideY = toksX * f, toksY * f

from_embed = perceptor.encode_text(clip.tokenize(from_text).to(device)).float()
to_embed = perceptor.encode_text(clip.tokenize(to_text).to(device)).float()

image = TF.to_tensor(pil_image.resize((sideX, sideY), Image.LANCZOS)).to(device).unsqueeze(0)
mask_dist = None
mask_total = 0.



# Are we using the mask we just generated?
if use_mask:
    if 'A' in pil_mask.getbands():
        pil_mask = pil_mask.getchannel('A')
    elif 'L' in pil_mask.getbands():
        pil_mask = pil_mask.getchannel('L')
    else:
        raise RuntimeError('Mask must have an alpha channel or be one channel')
    mask = TF.to_tensor(pil_mask.resize((toksX, toksY), Image.BILINEAR))
    mask = mask.to(device).unsqueeze(0)
    mask_dist = TF.to_tensor(pil_mask.resize((sideX, sideY), Image.BILINEAR)).to(device).unsqueeze(0)

    # Threshold on the average of the mask
    std, mean = torch.std_mean(mask_dist.view(-1)[torch.nonzero(mask_dist.view(-1))])
    std = std.item()
    mean = mean.item()
    # print(mean + (0.5) * std)
    mask = mask.lt(mean).float()

    if invert_mask:
        mask = 1 - mask
    mask_total = mask_dist.view(-1).sum()
else:
    mask = torch.ones([], device=device)

z = model.quant_conv(model.encoder(image * 2 - 1))
z.requires_grad_()
torch.set_grad_enabled(True)
opt = optim.Adam([z], lr=0.15)


# Draw picture
# Vector quantize
def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


def synth(z):
    if gumbel:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embed.weight).movedim(3, 1)
    else:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)


# def synth(z, sample=False):
#     logits = model.quantize.proj(z)
#     if sample:
#         one_hot = F.gumbel_softmax(logits, tau=1, hard=True, dim=1)
#     else:
#         one_hot = F.one_hot(logits.argmax(1), logits.shape[1]).movedim(3, 1).to(logits.dtype)
#     z_q = torch.einsum('nchw,cd->ndhw', one_hot, model.quantize.embed.weight)
#     return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

# Draw picture + print status + save picture
@torch.no_grad()
def checkin(i, losses):
    losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
    out = synth(z)
    TF.to_pil_image(out[0].cpu()).save('progress.png')
    # display.display(display.Image('progress.png'))


def ascend_txt():
    out = synth(z)
    seed = torch.randint(2 ** 63 - 1, [])

    noise_val = (1 - t) * 0.1

    # Random crops
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        out_embeds = perceptor.encode_image(normalize(make_cutouts(out))).float()

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        image_embeds = perceptor.encode_image(normalize(make_cutouts(image))).float()

    if mask_dist is not None:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            mask_scores = make_cutouts(mask_dist, augs=False, grads=False).view(cut_out_num, -1).sum(
                dim=-1) / mask_total

    result = []
    # Compare the image we started with to crops of the current image
    image_analogy = spherical_dist(out_embeds, image_embeds)  # * (torch.ones_like(mask_scores) - mask_scores)
    result.append(image_analogy.mean())
    # Move over a spherical geodesic that connects the "from state" to the "to state"
    word_analogy = (
                spherical_dist(out_embeds, to_embed, noise=False, noise_coeff=noise_val) - spherical_dist(out_embeds,
                                                                                                          from_embed,
                                                                                                          noise=False,
                                                                                                          noise_coeff=noise_val))
    result.append(word_analogy.mean() * scale_dir_by)

    return result  # Loss


def train(i):
    global t
    t = min(float(i) / float(cut_pow_length), 1.0)
    cur_cut_pow = (1 - t) * cut_pow_start + t * cut_pow_end
    make_cutouts.set_cut_pow(cur_cut_pow)

    # global scale_dir_by
    # scale_dir_by = clamp(1.0, sdbo * (1 - t) + t, sdbo)

    opt.zero_grad()
    lossAll = ascend_txt()
    if i % 50 == 0:
        checkin(i, lossAll)
    loss = sum(lossAll)
    loss.backward()
    opt.step()


i = 0
try:
    with tqdm() as pbar:
        while i < 500:
            train(i)
            i += 1
            pbar.update()
except KeyboardInterrupt:
    pass
