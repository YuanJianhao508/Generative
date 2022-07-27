import torch
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="2"

from utils import getMask
from generative_model import clip

generative_model = "VQGAN"

if generative_model == 'GLIDE':
    from glide_text2im.download import load_checkpoint
    from glide_text2im.model_creation import (
        create_model_and_diffusion,
        model_and_diffusion_defaults,
        model_and_diffusion_defaults_upsampler
    )
    from generative_model.GLIDE import glide_intervention

if generative_model == 'VQGAN':
    from generative_model.VQGAN import VQGAN_intervention, load_vqgan_model


has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')
print('Using device:', device)
torch.cuda.empty_cache()



checkpoint_dir = "/datasets/jianhaoy/checkpoints"
img_basedir = "/datasets/jianhaoy/celebA/img_align_celeba"

intervention_meta_dir = "/datasets/jianhaoy/celebA/img_align_celeba/interventionOutput/interventionMetadata"
meta_dir = "/datasets/jianhaoy/celebA/img_align_celeba"
use_mask = True  #@param {type:"boolean"}
invert_mask = True  #@param {type:"boolean"}
mask_samples =  1#@param {type:"integer"}


# Prepare CLIP Model
clip_model = "ViT-B/32"
perceptor, preprocess = clip.load(clip_model, jit=False)
perceptor.eval().requires_grad_(False).to(device)

# create generative model
# Create base model.
if generative_model == 'GLIDE':
    output_dir = "/datasets/jianhaoy/celebA/img_align_celeba/interventionOutput/interventionimage"
    options = model_and_diffusion_defaults()
    options['inpaint'] = True
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir,'base_inpaint.pt')))
    print('total base parameters', sum(x.numel() for x in model.parameters()))

    options_up = model_and_diffusion_defaults_upsampler()
    options_up['inpaint'] = True
    options_up['use_fp16'] = has_cuda
    options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cuda:
        model_up.convert_to_fp16()
    model_up.to(device)
    model_up.load_state_dict(torch.load(os.path.join(checkpoint_dir,'upsample_inpaint.pt')))
    print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

    # Sampling parameters
    # prompt = "sketch"
    batch_size = 1
    guidance_scale = 5.0
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    upsample_temp = 0.997


if generative_model == 'VQGAN':
    output_dir = "/datasets/jianhaoy/celebA/img_align_celeba/interventionOutput/VQGANimage"
    model = load_vqgan_model('/datasets/jianhaoy/checkpoints/vqgan_imagenet_f16_16384.yaml', '/datasets/jianhaoy/checkpoints/vqgan_imagenet_f16_16384.ckpt')
    model.to(device)



i = 0
metadata_df = pd.read_csv(os.path.join(intervention_meta_dir, "intervention_metadata.csv"))

#TODO quick test for waterbird need to make this a automatic process in pipeline later

# Use the same forma
fabricated_metadata = pd.DataFrame(columns=['img_id','img_filename','y','split','place','place_filename'])

# subject_mapping_dictionary = {'standing bird':0,'flying bird':1}
# background_mapping_dictionary = {'ocean':1,'forest':0,'lake':1,'bamboo leaf':0}

subject_mapping_dictionary = {'non-blond':0,'blond':1}
background_mapping_dictionary = {'woman':0,'man':1}

for i in tqdm(range(metadata_df.shape[0]),desc='Transform Intervention in Text Domain to Image Domain'):
    print("current image:",i)
    info = metadata_df.loc[i]
    img_path = info['img_path']
    img_path = os.path.join(img_basedir, img_path)
    subject = info['ClassLabel']
    prompt_list = info['InterventionDescription'].split("-")
    background = info['NonCausal']
    # origcap = info['OriginalDescription']
    # print(origcap)
    print(prompt_list)
            #get mask through CLIP
    if use_mask:
        mask = getMask.get_mask(perceptor,preprocess,subject,img_path,device=device,mask_samples=mask_samples)
    else:
        mask = torch.ones([], device=device)

    for prompt in tqdm(prompt_list):
        savename = savename = subject+str(i)+'-'+prompt+'.jpg'
        img_filename = os.path.join('interventionOutput\interventionimage',savename)
        if os.path.exists(os.path.join(output_dir,savename)):
            continue
        if subject_mapping_dictionary[subject] == 0:
            continue
        if generative_model == 'GLIDE':
            template = 'a {} of {} a {}'
            prompt = template.format("photo",subject,prompt)
            # Intervention Pregenerate all, since generative model is slow in run-time
            img_result = glide_intervention(img_path,mask,model,options,diffusion,model_up,options_up,diffusion_up,prompt,
                                                batch_size,guidance_scale,upsample_temp,device)
            img_result.save(os.path.join(output_dir,savename))


        if generative_model == 'VQGAN':
            template = 'a {} of {} a {}'
            from_text = info['OriginalDescription']
            to_text = template.format("photo",subject,prompt)
            img_result = VQGAN_intervention(img_path,mask,subject,from_text,to_text,perceptor,preprocess,model,device,use_mask=True,invert_mask=False,max_iter=300,image_size=224)
            img_result.save(os.path.join(output_dir,savename))




        # Create fabricated metadata
        size = fabricated_metadata.index.size
        fabricated_metadata.loc[size] = [1,img_filename,subject_mapping_dictionary[subject],0,background_mapping_dictionary[background],0]


original_metadata = pd.read_csv(os.path.join(meta_dir, "metadata.csv"))
fabricated_metadata = pd.concat([fabricated_metadata, original_metadata], ignore_index=True)
fabricated_metadata.to_csv(os.path.join(meta_dir,"fabricated_metadata.csv"),index=False)


