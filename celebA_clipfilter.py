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

basedir = "/datasets/jianhaoy/celebA/img_align_celeba/"
img_dir = "/datasets/jianhaoy/celebA/img_align_celeba/"
fabricated_metadata = pd.DataFrame(columns=['img_id','img_filename','y','split','place','place_filename'])
# files= os.listdir(img_dir) 
frac = 0.5
frac_lis = [0.3,0.4,0.5,0.6,0.7,0.8]
multi_frac = True
filter = True
upsample = False

if filter:
    model, preprocess = clip.load("ViT-B/32")
    style_classes,image_classes,background_classes,image_templates,rand_background = extract_template("celebA")
    print("1.Class Labels: ", image_classes)
    print("2.Backgrounds: ", background_classes)
    print("3.Style Labels: ", style_classes)
    print("4.templates:", image_templates)

    subject_mapping_dictionary = {0:'non-blond',1:'blond'}
    background_mapping_dictionary = {0:'woman',1:'man'}

    fabricated_metadata = pd.DataFrame(columns=['img_id','img_filename','y','split','place','score'])

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

old_f_meta = pd.read_csv("/datasets/jianhaoy/celebA/img_align_celeba/styleclipall1_metadata.csv")
orig_meta = pd.read_csv("/datasets/jianhaoy/celebA/img_align_celeba/metadata.csv")
for i in tqdm(range(old_f_meta.shape[0]-orig_meta.shape[0])):
    if  old_f_meta.iloc[i]['split'] != 0:
        continue
    # print(file)
    # filename = os.path.join('interventionOutput/interventionimage',file)
    filename = old_f_meta.iloc[i]['img_filename']
    y = old_f_meta.iloc[i]['y']
    place = old_f_meta.iloc[i]['place']
    # filename = os.path.join('interventionOutput/StyleCLIPimage',file)
    if y == 0:
        subject = 0
    else:
        subject = 1
    if place == 0:
        bg = 0
    else:
        bg = 1

    # if subject == 1 and bg == 1:
    #     print(filename)
    
    # If use CLIP to select sample:
    if filter:
        text = image_templates[0].format("photo",subject_mapping_dictionary[subject],background_mapping_dictionary[bg])
        # print(text)
        text_embedding = zeroshot_dict[text]
        img_path = os.path.join(img_dir,filename)
        img = Image.open(img_path).convert("RGB")
        images = transform(img).unsqueeze(0)
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        score = image_features @ text_embedding
        score = score.cpu().detach().numpy()
        
        # print(score)
        size = fabricated_metadata.index.size
        fabricated_metadata.loc[size] = [1, filename, subject, 0, bg, score]

    else:
        size = fabricated_metadata.index.size
        fabricated_metadata.loc[size] = [1, filename, subject, 0,
                                        bg]
    # if i > 100:
    #     break

if filter:
    print("filtering with CLIP:")
    fabricated_metadata.sort_values(by="score",inplace=True,ascending=False)
    nrow = fabricated_metadata.shape[0]

original_metadata = pd.read_csv(os.path.join(basedir, "metadata.csv"))

if multi_frac:
    for i in frac_lis:
        fabricated_metadata_f = fabricated_metadata[0:round(nrow*i)]
        # print(fabricated_metadata.head())
        fabricated_metadata_f.drop(columns='score',inplace=True)
        fabricated_metadata_f = pd.concat([fabricated_metadata_f, original_metadata], ignore_index=True)
        fabricated_metadata_f.to_csv(os.path.join(basedir,f"styleclip{i}.csv"),index=False)
else:
    fabricated_metadata_f = fabricated_metadata[0:round(nrow*frac)]
    # print(fabricated_metadata.head())
    fabricated_metadata_f.drop(columns='score',inplace=True)
    fabricated_metadata_f = pd.concat([fabricated_metadata_f, original_metadata], ignore_index=True)
    fabricated_metadata_f.to_csv(os.path.join(basedir,f"styleclip{frac}.csv"),index=False)



# if upsample:
#     final_metadata = pd.concat([fabricated_metadata, fabricated_metadata], ignore_index=True)
#     fabricated_metadata = pd.concat([fabricated_metadata, final_metadata], ignore_index=True)


# original_metadata = pd.read_csv(os.path.join(basedir, "metadata.csv"))
# fabricated_metadata = pd.concat([fabricated_metadata, original_metadata], ignore_index=True)
# fabricated_metadata.to_csv(os.path.join(basedir,"styleclipf5.csv"),index=False)

