# Import standard libraries
import os
import pickle
import joblib
import argparse
import logging

# Import data and visualization libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import wandb

# Add local utility functions to the system path
import sys

sys.path.insert(0, "./utils/")
import model_utils
import embed_utils
import eval_utils

# Config parsing
import json
import yaml

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

# Suppress warnings
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


def main(args):
    # Get current working directory
    cwd = os.getcwd()

    # Load main config and model-specific config, and merge them
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    with open(args.model_config, "r") as stream:
        config.update(yaml.safe_load(stream))

    # Add additional parameters from command-line arguments
    config["iso_code"] = args.iso_code
    config["mode"] = args.mode

    # Extract relevant parameters from config
    target_columns = config[args.mode]
    embed_model_name = config["embed_model"]
    model_name = config["model"]
    path_to_images = config["path_to_images"].format(
        mode=args.mode, iso_code=args.iso_code
    )
    path_to_embeddings = config["path_to_embeddings"].format(
        mode=args.mode, iso_code=args.iso_code
    )
    path_to_file = config["path_to_file"].format(mode=args.mode, iso_code=args.iso_code)

    # Load the geospatial data, replace 'nan' strings, and filter rows
    raw_data = gpd.read_file(path_to_file)
    data = raw_data.copy().dropna()
    data = data[(data.duplicate == False) & (data.clean == True)]
    data = data.reset_index()

    # Generate full image file paths
    data["filepath"] = data.filename.apply(lambda x: os.path.join(path_to_images, x))

    # Log dataset size and distribution of target labels
    logging.info(f"Data dimensions: {data.shape[0]}")
    for target_column in target_columns:
        logging.info(data[target_column].value_counts())

    # Generate image embeddings using specified model
    embeddings = embed_utils.generate_embeddings(
        data=data,
        image_dir=path_to_images,
        out_dir=path_to_embeddings,
        model_name=embed_model_name,
    ).reset_index(drop=True)

    # Ensure embedding column names are strings
    embeddings.columns = [str(x) for x in embeddings.columns]
    n_features = len(embeddings.columns) - 1

    # Filter embeddings to only include data points present in `data`
    embeddings = embeddings[embeddings.UID.isin(data.UID.unique())]
    embeddings = embeddings.merge(data, on="UID", how="inner")

    # Loop through each target column to train a separate model
    for target_column in target_columns:
        group_column = config["group_column"]  # Used for grouped cross-validation
        feature_columns = [str(x) for x in range(n_features)]

        # Define features, target variable, and groups
        X = embeddings[feature_columns]
        y = embeddings[target_column]
        groups = embeddings[group_column]

        dataset_col = f"{target_column}_dataset"
        if dataset_col not in data.columns:
            raw_data[dataset_col] = None
            train_index, test_index = model_utils.train_test_split(
                config, X, y, groups, test_size=config["test_size"]
            )
            raw_data.loc[
                raw_data.UID.isin(data.loc[train_index].UID.unique()), dataset_col
            ] = "train"
            raw_data.loc[
                raw_data.UID.isin(data.loc[test_index].UID.unique()), dataset_col
            ] = "test"
            raw_data.to_file(path_to_file, driver="GeoJSON")
        else:
            train_index = data.index[data[dataset_col] == "train"].tolist()
            test_index = data.index[data[dataset_col] == "test"].tolist()

        # Train model using group k-fold cross-validation
        cv, result, report, cm, preds = model_utils.group_kfold(
            config, X, y, groups, train_index, test_index
        )
        result = {f"test_{key}": val for key, val in result.items()}

        # Output evaluation metrics
        logging.info(result)
        logging.info(report)

        # Set up experiment directory for outputs
        exp_name = (
            f"{target_column.lower()}-{embed_model_name}-{model_name}-{args.iso_code}"
        )
        exp_dir = os.path.join(cwd, f"exp/{args.iso_code}/{exp_name}")
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        # Save predictions and confusion matrix
        result["target"] = target_column
        eval_utils.save_results(preds, cm, result, report, exp_dir, exp_name)

        # Log results to Weights & Biases
        run = wandb.init(project=config["project"], config=config)
        run.name = exp_name
        wandb_report = wandb.Table(dataframe=report)
        run.log(result)
        run.log({"report": wandb_report})
        run.finish()

        # Print and save the best trained model
        logging.info(cv.best_estimator_)
        model = cv.best_estimator_.fit(X, y)
        joblib.dump(model, os.path.join(exp_dir, f"{exp_name}.joblib"))


# Entry point when running as script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--iso_code", help="ISO code", default="VCT")
    parser.add_argument("--mode", help="Aerial or streetview", default="aerial")
    parser.add_argument("--config", help="Main config", default="configs/config.yaml")
    parser.add_argument(
        "--model_config",
        help="Model config",
        default="configs/ml_configs/ResNet50_FMOW_RGB_GASSL-MLP.yaml",
    )
    args = parser.parse_args()

    main(args)
