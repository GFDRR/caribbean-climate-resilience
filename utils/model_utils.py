import os
import rasterio as rio
import torch
import cv2
import numpy as np
import torchvision
import timm
import matplotlib.pyplot as plt
import torch.nn as nn

from tqdm import tqdm
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
from torchgeo.models import DOFABase16_Weights, dofa_base_patch16_224
from torchgeo.models import Swin_V2_B_Weights, swin_v2_b
from torchgeo.models import ResNet50_Weights

import satlaspretrain_models

NAIP_MEAN = [123.675, 116.28, 103.53] 
NAIP_STD = [58.395, 57.12, 57.375] 
NAIP_WAVELENGTHS = [0.665, 0.56, 0.49]

def top_n_similarity(vector, vector_list, indexes, n=5):
    """
    Calculates cosine similarity between a vector and a list of vectors,
    and returns the top n most similar vectors.

    Args:
        vector (np.ndarray): The query vector.
        vector_list (np.ndarray): A list of vectors to compare against.
        n (int): The number of top similar vectors to return.

    Returns:
        list: A list of tuples, where each tuple contains the index and cosine 
              similarity score of the top n most similar vectors.
    """
    similarity_scores = cosine_similarity([vector], vector_list)[0]
    
    # Create a list of (index, similarity) tuples
    scored_ids = list(zip(indexes, similarity_scores))

    # Sort and select top n
    top_similar = sorted(scored_ids, key=lambda x: x[-1], reverse=True)[:n]

    return top_similar


def load_model_transform(model_name: str, image_size=224):
    if model_name == "vit_base_dofa":
        model = dofa_base_patch16_224(weights=DOFABase16_Weights.DOFA_MAE)
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((image_size, image_size)), 
            transforms.CenterCrop(image_size), 
            transforms.Normalize(mean=NAIP_MEAN, std=NAIP_STD)
        ])

    elif model_name == "Aerial_SwinB_SI":
        weights = Swin_V2_B_Weights.NAIP_RGB_MI_SATLAS
        model = swin_v2_b(weights)
        model = nn.Sequential(*list(model.children())[:-1])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)), 
            transforms.CenterCrop(image_size)
        ])

    elif model_name == "FMOW_RGB_GASSL":
        weights = ResNet50_Weights.FMOW_RGB_GASSL
        in_chans = weights.meta['in_chans']
        model = timm.create_model('resnet50')
        model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((image_size, image_size)), 
            transforms.CenterCrop(image_size), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        model = torch.nn.Sequential(*list(model.children())[:-1])
        
    model = model.cuda()
    model.eval()
    return model, transform


def load_image(image_file, padding_color=(0, 0, 0)):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape
    max_size = max(height, width)
    
    pad_top = (max_size - height) // 2
    pad_bottom = max_size - height - pad_top
    pad_left = (max_size - width) // 2
    pad_right = max_size - width - pad_left

    # Pad the image
    image = cv2.copyMakeBorder(
        image, 
        pad_top, 
        pad_bottom, 
        pad_left, 
        pad_right,
        cv2.BORDER_CONSTANT, 
        value=padding_color
    )
    #image = image / 255
    return image


def generate_embedding(
    data: pd.DataFrame,
    image_dir: str,
    out_dir: str,
    model_name: str = "FMOW_RGB_GASSL"# "Aerial_SwinB_SI"
):
    filename = os.path.join(out_dir, f"{model_name}.csv")
    if os.path.exists(filename):
        embeddings = pd.read_csv(filename)
        return embeddings
    
    model, transform = load_model_transform(model_name)
    
    embeddings = []
    data = data.reset_index(drop=True)
    for index in tqdm(range(len(data)), total=len(data)):
        item = data.iloc[index]
        image = load_image(item.filepath)
        image = transform(image)[:3].unsqueeze(0)
            
        if model_name == "vit_base_dofa":
            out_feat = model.forward_features(image.to(torch.float32).cuda(), NAIP_WAVELENGTHS)
        else:
            out_feat = model(image.to(torch.float32).cuda())

        embedding = list(np.array(out_feat[0].cpu().detach().numpy()).reshape(1, -1)[0])
        embeddings.append(embedding)

    embeddings = pd.DataFrame(embeddings, columns = [str(x) for x in range(len(embeddings[0]))]) 
    embeddings["UID"] = data.UID
    
    os.makedirs(out_dir, exist_ok=True)
    embeddings.to_csv(filename, index=False)
    
    return embeddings