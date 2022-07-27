import numpy as np
import torch
import clip
from tqdm import tqdm
import argparse
import random
import copy
import pandas as pd
from dataset.wb_data import WaterBirdsDataset, get_loader, get_transform_cub
from utils.get_template import extract_template
import os

parser = argparse.ArgumentParser(description="Generate Intervention Text Prompt with Zero-shot CLIP:")
parser.add_argument(
    "--dataset", type=str,
    default=None,
    help="Using What?")
parser.add_argument(
    "--basedir", type=str,
    default=None,
    help="Image/Metadata Directory")
parser.add_argument(
    "--outputdir", type=str,
    default=None,
    help="Intervention token output (save csv)")

args = parser.parse_args()

#Get all possible templates and calcuate zero-shot similarity in parallel (speed up) adapted CLIP
def zeroshot_classifier(classname,style_classes,background_classes,templates):
    with torch.no_grad():
        zeroshot_weights = []
        descriptions = []
        for background_class in background_classes:
            for style_class in style_classes:
                for template in templates:
                    texts = [template.format(style_class,classname,background_class)] #format with class
                    # style, class label, background, caption template
                    elements = [style_class,classname,background_class,template]
                    descriptions.append(elements)
                    texts = clip.tokenize(texts).cuda() #tokenize
                    class_embeddings = model.encode_text(texts) #embed with text encoder
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights,descriptions

def zeroshot_class_template(image_classes):
    zeroshot_weight_dictionary = {}
    description_dictionary = {}
    for classname in tqdm(image_classes):
        zeroshot_weight,description = zeroshot_classifier(classname,style_classes,background_classes, image_templates)
        zeroshot_weight_dictionary.update({classname:zeroshot_weight})
        description_dictionary.update({classname:description})
    return zeroshot_weight_dictionary,description_dictionary

def get_description(output,class_descriptions):
    #topk [1] get index and tranfer to tensor
    pred = output.topk(1, 1, True, True)[1].t()
    pred = class_descriptions[pred]
    return pred


# Main
model, preprocess = clip.load("ViT-B/32")


#class label and prompt template
n_intervention_bg_samples = 3
n_intervention_st_samples = 0
style_classes, image_classes,background_classes,image_templates,rand_background = extract_template(args.dataset)
print("1.Class Labels: ", image_classes)
print("2.Backgrounds: ", background_classes)
print("3.Style Labels: ", style_classes)
print("4.templates:", image_templates)


zeroshot_weight_dictionary,description_dictionary = zeroshot_class_template(image_classes)
print(description_dictionary)

#Data: This is to generate all the image descriptions and intervention on text domain for traing set
target_resolution = (224, 224)
transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=False)
trainset = WaterBirdsDataset(basedir=args.basedir, split="train", transform=transform, intervention=True)
loader_kwargs = {'batch_size': 1,'num_workers': 4,'pin_memory': True}
train_loader = get_loader(trainset, train=False, **loader_kwargs)
target_map = trainset.get_target_mapping(args.dataset)
intervention_metadata = pd.DataFrame(columns=['ClassLabel','OriginalDescription','InterventionDescription','img_path','NonCausal','background_aug'])
correlation_shift = True
domain_shift = False

i = 0
with torch.no_grad():
    top1,  n = 0., 0.
    i = 0
    for batch in tqdm(train_loader):
        img, label, background, path = batch
        images = img.cuda()
        target = label
        target = target_map[int(target)]
        # print(target)
        class_weights = zeroshot_weight_dictionary[target]
        class_descriptions = description_dictionary[target]
        # print(class_descriptions)

        # PredictOriginalDescription
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100. * image_features @ class_weights
        pred = get_description(logits,class_descriptions)
        # print(pred)

        #Intervention on Background and Style in Text Domain
        if correlation_shift:
            temp_back = copy.deepcopy(background_classes)
            temp_back.remove(pred[2])
            intervention_background_tokens = random.sample(temp_back, n_intervention_bg_samples)
            rand_adj_token = random.choices(rand_background,k=n_intervention_bg_samples)
            intervention_samples = '-'.join(intervention_background_tokens)
        if domain_shift:
            temp_style = copy.deepcopy(style_classes)
            temp_style.remove(pred[0])
            intervention_style_tokens = random.sample(temp_back, n_intervention_st_samples)
            intervention_samples += '-'.join(intervention_style_tokens)

        orig_caption = [pred[3].format(pred[0], target, pred[2])]
        # Create Fake metadata:
        size = intervention_metadata.index.size
        intervention_metadata.loc[size] = [label, orig_caption[0], intervention_samples, path[0],pred[2],rand_adj_token]
        
    intervention_metadata.to_csv(os.path.join(args.outputdir,"intervention_4bg_metadata.csv"),index=False)





