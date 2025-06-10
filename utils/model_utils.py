import os
import cv2
import torch
import numpy as np
import pandas as pd
import rasterio as rio
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

import os
from tqdm import tqdm
import collections
import itertools
import random
import copy

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timm

from sklearn.preprocessing import (
    MinMaxScaler,
    MaxAbsScaler,
    StandardScaler,
    RobustScaler,
)

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    GridSearchCV,
    RandomizedSearchCV,
    GroupShuffleSplit,
)

import cv2
import torch
import rasterio as rio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from torchvision.transforms import ToPILImage
from torchvision import transforms

import sys

sys.path.insert(0, "./utils/")
import eval_utils
import clf_utils

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
from torchgeo.models import (
    DOFABase16_Weights,
    dofa_base_patch16_224,
    Swin_V2_B_Weights,
    swin_v2_b,
    ResNet50_Weights,
    ScaleMAELarge16_Weights,
    scalemae_large_patch16,
)

SEED = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NAIP_MEAN = [123.675, 116.28, 103.53]
NAIP_STD = [58.395, 57.12, 57.375]
NAIP_WAVELENGTHS = [0.665, 0.56, 0.49]


def _get_scalers(scalers):
    """
    Returns a list of scalers for hyperparameter optimization.

    Args:
        scalers (list): A list of strings indicating the scalers
            to include in the hyperparameter search space.

    Returns:
        list: A list of sclaer instances.
    """

    scalers_list = [None]

    for scaler in scalers:
        scalers_list.append(clf_utils.get_scaler(scaler))

    return scalers_list


def _get_pipeline(model, selector):
    """
    Instantiates and returns a pipeline based on the input configuration.

    Args:
        model (object): The model instance to include in the pipeline.
        selector (object): The selector instance to include in the pipeline.

    Returns:
        sklearn pipeline instance.
    """

    if model in clf_utils.MODELS:
        model = clf_utils.get_model(model)

    if selector in clf_utils.SELECTORS:
        selector = clf_utils.get_selector(selector)

    return Pipeline(
        [
            ("scaler", "passthrough"),
            ("selector", selector),
            ("model", model),
        ]
    )


def _get_params(scalers, model_params, selector_params):
    """
    Instantiates the parameter grid for hyperparameter optimization.

    Args:
        scalers (dict): A dictionary indicating the the list of scalers.
        model_params (dict): A dictionary containing the model parameters.
        selector_params (dict): A dictionary containing the feature
            selector parameters.

    Returns
        dict: Contains the parameter grid, combined into a single dictionary.
    """

    def _get_range(param):
        if param[0] == "np.linspace":
            return list(np.linspace(*param[1:]).astype(int))
        elif param[0] == "range":
            return list(range(*param[1:]))
        return param

    scalers = {"scaler": _get_scalers(scalers)}

    if model_params:
        model_params = {
            "model__" + name: _get_range(param) for name, param in model_params.items()
        }
    else:
        model_params = {}

    if selector_params:
        selector_params = {
            "selector__" + name: _get_range(param)
            for name, param in selector_params.items()
        }
    else:
        selector_params = {}

    params = [model_params, selector_params, scalers]

    return dict(collections.ChainMap(*params))


def get_cv(c, cv):
    """
    Returns a model selection instance.

    Args:
        c (dict): The config dictionary indicating the model,
            selector, scalers, parameters, and model selection
            instance.

    Returns:
        object: The model selector instance.
    """

    pipe = _get_pipeline(c["model"], c["selector"])
    params = _get_params(c["scalers"], c["model_params"], c["selector_params"])
    cv_strategy, cv_params = c["cv"], c["cv_params"]

    assert cv_strategy in ["RandomizedSearchCV", "GridSearchCV"]

    scoring = eval_utils.get_scoring()
    if cv_strategy == "RandomizedSearchCV":
        return RandomizedSearchCV(
            pipe, params, scoring=scoring, random_state=SEED, cv=cv, **cv_params
        )
    elif cv_strategy == "GridSearchCV":
        return GridSearchCV(pipe, params, scoring=scoring, cv=cv, **cv_params)


def group_kfold(c, X, y, groups, test_size=0.2):
    """
    Performs grouped cross-validation with GroupKFold after an initial train-test split using GroupShuffleSplit.

    Args:
        c (dict): Configuration dictionary, must include 'n_splits' for cross-validation.
        X (pd.DataFrame): Feature dataframe.
        y (pd.Series or np.ndarray): Target variable.
        groups (pd.Series or np.ndarray): Group labels for the samples used while splitting the dataset.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.

    Returns:
        tuple:
            cv (sklearn.model_selection.GridSearchCV or similar): Fitted cross-validator with best model.
            result (dict): Evaluation metrics computed by `eval_utils.evaluate`.
            report (pd.DataFrame): Classification report as a pandas DataFrame.
            cm (sklearn.metrics.ConfusionMatrixDisplay): Confusion matrix display object.
            preds (pd.DataFrame): DataFrame with columns `y_pred` and `y_test` containing predictions and ground truth.
    """
    gss = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=42)
    split = gss.split(X, y, groups=groups)
    train_index, test_index = next(split)

    (
        X_train,
        X_test,
    ) = (
        X.loc[train_index],
        X.loc[test_index],
    )
    y_train, y_test = y[train_index], y[test_index]
    group_train, group_test = groups[train_index], groups[test_index]

    cv = get_cv(c, GroupKFold(c["n_splits"]))
    cv.fit(X_train, y_train, groups=group_train)

    cv_results = pd.DataFrame.from_dict(cv.cv_results_)
    cv_results = cv_results.sort_values("mean_test_f1_score", ascending=False)

    y_pred = cv.best_estimator_.predict(X_test)
    result = eval_utils.evaluate(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    report = pd.DataFrame(report).transpose()

    labels = list(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    preds = pd.DataFrame.from_dict({"y_pred": y_pred, "y_test": y_test})

    return cv, result, report, cm, preds


def top_n_similarity(
    vector: np.ndarray, vector_list: np.ndarray, indexes: list, n: int = 25
):
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
        model = timm.create_model("resnet50")
        model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
        model = torch.nn.Sequential(*list(model.children())[:-1])

    elif model_name == "ScaleMAE_FMOW_RGB":
        weights = ScaleMAELarge16_Weights.FMOW_RGB
        model = scalemae_large_patch16(weights)

    model = model.cuda()
    model.eval()
    return model


def get_transform(model_name: str, image_size: int = 224):
    """
    Return the appropriate image transformation for a given model.

    Args:
        model_name (str): Model name for which to get the transform.
        image_size (int, optional): Target image size. Defaults to 224.

    Returns:
        torchvision.transforms.Compose: Composed transformation pipeline.
    """
    if model_name == "vit_base_dofa":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop(image_size),
                transforms.Normalize(mean=NAIP_MEAN, std=NAIP_STD),
            ]
        )

    elif model_name == "Aerial_SwinB_SI":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop(image_size),
            ]
        )

    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop(image_size),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


def load_image(image_file: str, padding_color: tuple = (0, 0, 0)):
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
        value=padding_color,
    )
    return image


def generate_embeddings(
    data: pd.DataFrame, image_dir: str, out_dir: str, model_name: str
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
        if len(embeddings) >= len(data):
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
            out_feat = model.forward_features(
                image.to(torch.float32).cuda(), NAIP_WAVELENGTHS
            )
            out_feat = model.forward_head(out_feat, pre_logits=True)
        elif model_name == "ScaleMAE_FMOW_RGB":
            unpool_features = model.forward_features(image.to(torch.float32).cuda())
            out_feat = model.forward_head(unpool_features, pre_logits=True)
        else:
            out_feat = model(image.to(torch.float32).cuda())

        embedding = list(np.array(out_feat[0].cpu().detach().numpy()).reshape(1, -1)[0])
        embeddings.append(embedding)
        torch.cuda.empty_cache()

    embeddings = pd.DataFrame(
        embeddings, columns=[str(x) for x in range(len(embeddings[0]))]
    )
    embeddings["UID"] = data.UID

    os.makedirs(out_dir, exist_ok=True)
    embeddings.to_csv(filename, index=False)

    return embeddings
