import os
import cv2
import torch
import numpy as np
import pandas as pd
import rasterio as rio
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
from torchgeo.models import (
    DOFABase16_Weights, dofa_base_patch16_224,
    Swin_V2_B_Weights, swin_v2_b,
    ResNet50_Weights,
    ScaleMAELarge16_Weights, scalemae_large_patch16
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NAIP_MEAN = [123.675, 116.28, 103.53] 
NAIP_STD = [58.395, 57.12, 57.375] 
NAIP_WAVELENGTHS = [0.665, 0.56, 0.49]

def top_n_similarity(vector, vector_list, indexes, n=5):
    """
    Calculate the top-N cosine similarities between a given vector and a list of vectors.

    Args:
        vector (np.ndarray): The query vector.
        vector_list (np.ndarray): The list of vectors to compare the query vector against.
        indexes (list): A list of identifiers corresponding to each vector in `vector_list`.
        n (int, optional): The number of top similar vectors to return. Defaults to 5.

    Returns:
        list: A list of tuples where each tuple contains an index and the corresponding 
              cosine similarity score, sorted in descending order of similarity.
    """
    similarity_scores = cosine_similarity([vector], vector_list)[0]
    scored_ids = list(zip(indexes, similarity_scores))
    top_similar = sorted(scored_ids, key=lambda x: x[-1], reverse=True)[:n]
    return top_similar


def load_model(model_name: str):
    """
    Load a pretrained vision model for embedding extraction.

    Args:
        model_name (str): Name of the model to load.
    Returns:
        torch.nn.Module: The loaded model in evaluation mode on CUDA.
    """
    if model_name == "vit_base_dofa":
        model = dofa_base_patch16_224(weights=DOFABase16_Weights.DOFA_MAE)

    elif model_name == "Aerial_SwinB_SI":
        weights = Swin_V2_B_Weights.NAIP_RGB_MI_SATLAS
        model = swin_v2_b(weights)
        model = nn.Sequential(*list(model.children())[:-1])

    elif model_name == "ResNet50_FMOW_RGB_GASSL":
        weights = ResNet50_Weights.FMOW_RGB_GASSL
        model = timm.create_model('resnet50')
        model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
        model = torch.nn.Sequential(*list(model.children())[:-1])

    elif model_name == "ScaleMAE_FMOW_RGB":
        weights = ScaleMAELarge16_Weights.FMOW_RGB
        model = scalemae_large_patch16(weights)
        
    model = model.cuda()
    model.eval()
    return model


def get_transform(model_name: str, image_size=224):
    """
    Return the appropriate image transformation for a given model.

    Args:
        model_name (str): Model name for which to get the transform.
        image_size (int, optional): Target image size. Defaults to 224.

    Returns:
        torchvision.transforms.Compose: Composed transformation pipeline.
    """
    if model_name == "vit_base_dofa":
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((image_size, image_size)), 
            transforms.CenterCrop(image_size), 
            transforms.Normalize(mean=NAIP_MEAN, std=NAIP_STD)
        ])

    elif model_name == "Aerial_SwinB_SI":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)), 
            transforms.CenterCrop(image_size)
        ])

    elif model_name == "ResNet50_FMOW_RGB_GASSL":
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((image_size, image_size)), 
            transforms.CenterCrop(image_size), 
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        model = torch.nn.Sequential(*list(model.children())[:-1])

    elif model_name == "ScaleMAE_FMOW_RGB":
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((image_size, image_size)), 
            transforms.CenterCrop(image_size), 
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        

def load_image(image_file, padding_color=(0, 0, 0)):
    """
    Load an image and pad it to make it square.

    Args:
        image_file (str): Path to the image file.
        padding_color (tuple, optional): RGB values for padding. Defaults to black.

    Returns:
        np.ndarray: Padded RGB image.
    """
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
    return image


def generate_embeddings(
    data: pd.DataFrame,
    image_dir: str,
    out_dir: str,
    model_name: str 
):
    """
    Generate and save image embeddings for a dataset.

    Args:
        data (pd.DataFrame): DataFrame with at least a 'filepath' and 'UID' column.
        image_dir (str): Directory where images are located.
        out_dir (str): Directory to save output embeddings.
        model_name (str): Name of the model to use for embedding generation.

    Returns:
        pd.DataFrame: DataFrame containing embeddings and corresponding UIDs.
    """
    filename = os.path.join(out_dir, f"{model_name}.csv")
    if os.path.exists(filename):
        embeddings = pd.read_csv(filename)
        return embeddings
    
    model = load_model(model_name)
    transform = get_transform(model_name)
    
    embeddings = []
    data = data.reset_index(drop=True)
    for index in tqdm(range(len(data)), total=len(data)):
        item = data.iloc[index]
        image = load_image(item.filepath)
        image = transform(image)[:3].unsqueeze(0)
            
        if model_name == "vit_base_dofa":
            out_feat = model.forward_features(image.to(torch.float32).cuda(), NAIP_WAVELENGTHS)
            out_feat = model.forward_head(unpool_features, pre_logits=True)
        elif model_name == "ScaleMAE_FMOW_RGB":
            unpool_features = model.forward_features(image.to(torch.float32).cuda())
            out_feat = model.forward_head(unpool_features, pre_logits=True)
        else:
            out_feat = model(image.to(torch.float32).cuda())

        embedding = list(np.array(out_feat[0].cpu().detach().numpy()).reshape(1, -1)[0])
        embeddings.append(embedding)
        torch.cuda.empty_cache()

    embeddings = pd.DataFrame(embeddings, columns = [str(x) for x in range(len(embeddings[0]))]) 
    embeddings["UID"] = data.UID
    
    os.makedirs(out_dir, exist_ok=True)
    embeddings.to_csv(filename, index=False)
    
    return embeddings