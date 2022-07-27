import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pandas as pd
import numpy as np
from get_template import extract_template
from PIL import Image
import clip
import torch
from tqdm import tqdm
import torchvision.transforms as transforms

has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')
print('Using device:', device)
torch.cuda.empty_cache()

basedir = "/datasets/jianhaoy/waterbird"
img_dir = "/datasets/jianhaoy/waterbird/interventionOutput/VQGANimage"
fabricated_metadata = pd.DataFrame(columns=['img_id','img_filename','y','split','place','place_filename'])
files= os.listdir(img_dir) 
frac = 0.9
filter = True
upsample = False

if filter:
    model, preprocess = clip.load("ViT-B/32")
    style_classes,image_classes,background_classes,image_templates,rand_background = extract_template("waterbird")
    print("1.Class Labels: ", image_classes)
    print("2.Backgrounds: ", background_classes)
    print("3.Style Labels: ", style_classes)
    print("4.templates:", image_templates)

    subject_mapping_dictionary = {0:'standing bird',1:'flying bird'}
    background_mapping_dictionary = {1:'ocean',0:'forest'}

    fabricated_metadata = pd.DataFrame(columns=['img_id','img_filename','y','split','place','place_filename','score'])

    def zeroshot_classifier(classnames,style_classes,background_classes,templates):
        with torch.no_grad():
            zeroshot_weights = []
            descriptions = []
            dic = {}
            for classname in classnames:
                for background_class in background_classes:
                    for style_class in style_classes:
                        for template in templates:
                            text = template.format(style_class,classname,background_class) #format with class
                            texts = clip.tokenize(text).cuda() #tokenize
                            class_embeddings = model.encode_text(texts) #embed with text encoder
                            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                            class_embedding = class_embeddings.mean(dim=0)
                            class_embedding /= class_embedding.norm()
                            dic.update({text:class_embedding})
        return dic

    zeroshot_dict = zeroshot_classifier(image_classes,style_classes,background_classes, image_templates)

    scale = 256.0 / 224.0
    target_resolution = (224,224)
    transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


for file in tqdm(files): 
    # print(file)
    # filename = os.path.join('interventionOutput/interventionimage',file)
    filename = os.path.join('interventionOutput/VQGANimage',file)
    if "stand" in file:
        subject = 0
    else:
        subject = 1
    if "ocean" in file:
        bg = 1
    else:
        bg = 0
    # If use CLIP to select sample:
    if filter:
        text = image_templates[0].format("photo",subject_mapping_dictionary[subject],background_mapping_dictionary[bg])
        text_embedding = zeroshot_dict[text]
        img_path = os.path.join(img_dir,file)
        img = Image.open(img_path).convert("RGB")
        images = transform(img).unsqueeze(0)
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        score = image_features @ text_embedding
        score = score.cpu().detach().numpy()
        
        size = fabricated_metadata.index.size
        fabricated_metadata.loc[size] = [1, filename, subject, 0, bg, 0, score]

    else:
        size = fabricated_metadata.index.size
        fabricated_metadata.loc[size] = [1, filename, subject, 0,
                                        bg, 0]

if filter:
    print("filtering with CLIP:")
    fabricated_metadata.sort_values(by="score",inplace=True,ascending=False)
    nrow = fabricated_metadata.shape[0]

    fabricated_metadata = fabricated_metadata[0:round(nrow*frac)]
    fabricated_metadata.drop(columns='score',inplace=True)



if upsample:
    final_metadata = pd.concat([fabricated_metadata, fabricated_metadata], ignore_index=True)
    fabricated_metadata = pd.concat([fabricated_metadata, final_metadata], ignore_index=True)


original_metadata = pd.read_csv(os.path.join(basedir, "metadata.csv"))
fabricated_metadata = pd.concat([fabricated_metadata, original_metadata], ignore_index=True)
fabricated_metadata.to_csv(os.path.join(basedir,"vqganf9_metadata.csv"),index=False)

