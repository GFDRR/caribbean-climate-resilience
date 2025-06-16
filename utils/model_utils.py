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

from sklearn.preprocessing import (
    MinMaxScaler,
    MaxAbsScaler,
    StandardScaler,
    RobustScaler,
)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    GridSearchCV,
    RandomizedSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
)

import cv2
import torch
import rasterio as rio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, "./utils/")
import eval_utils
import clf_utils

from tqdm import tqdm
import torchvision.transforms as transforms

SEED = 42


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


def group_kfold(c, X, y, groups, train_index, test_index):
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
    (
        X_train,
        X_test,
    ) = (
        X.loc[train_index],
        X.loc[test_index],
    )
    y_train, y_test = y[train_index], y[test_index]
    print("Train")
    print(y_train.value_counts())
    print("Test")
    print(y_test.value_counts())
    group_train, group_test = groups[train_index], groups[test_index]

    cv = get_cv(c, GroupKFold(c["n_splits"]))
    cv.fit(X_train, y_train, groups=group_train)

    cv_results = pd.DataFrame.from_dict(cv.cv_results_)
    cv_results = cv_results.sort_values("mean_test_f1_score", ascending=False)

    y_pred = cv.best_estimator_.predict(X_test)
    result, report = eval_utils.evaluate(y_test, y_pred)

    labels = list(y_test.unique())
    cm, cm_display, cm_metrics = eval_utils.get_confusion_matrix(y_test, y_pred, labels)
    preds = pd.DataFrame.from_dict({"y_pred": y_pred, "y_test": y_test})

    return cv, result, report, cm_display, preds


def train_test_split(c, X, y, groups, test_size=0.2):
    def StratifiedGroupShuffleSplit(
        X, y=None, groups=None, test_size: float = 0.25, random_state=None
    ):
        desired = 1.0 / test_size
        c, f = np.ceil(desired), np.floor(desired)
        n_folds = int(c if c / desired < desired / f else f)
        return StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )

    # gss = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=SEED)
    gss = StratifiedGroupShuffleSplit(X, y, groups, test_size, random_state=SEED)
    split = gss.split(X, y, groups=groups)
    train_index, test_index = next(split)
    return train_index, test_index
