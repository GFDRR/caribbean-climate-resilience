import os 
import pickle 
import joblib 
import argparse
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import wandb

import sys
sys.path.insert(0, "./utils/")
import model_utils
import json
import yaml

logging.basicConfig(level=logging.INFO)

# Suppress warnings
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


def main(args):
    cwd = os.getcwd()
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    with open(args.model_config, 'r') as stream:
        config.update(yaml.safe_load(stream))

    config["iso_code"] = args.iso_code
    config["mode"] = args.mode
    
    target_columns = config[args.mode]
    embed_model_name = config["embed_model"]
    model_name = config["model"]
    path_to_images = config['path_to_images'].format(mode=args.mode, iso_code=args.iso_code)
    path_to_embeddings = config['path_to_embeddings'].format(mode=args.mode, iso_code=args.iso_code)
    path_to_file = config['path_to_file'].format(mode=args.mode, iso_code=args.iso_code)

    data = gpd.read_file(path_to_file).replace('nan', np.nan).dropna()
    data = data[(data.duplicate == False) & (data.clean == True)]
    data["filepath"] = data.filename.apply(lambda x: os.path.join(path_to_images, x))
    logging.info(f"Data dimensions: {data.shape[0]}")
    for target_column in target_columns:
        logging.info(data[target_column].value_counts())
        
    embeddings = model_utils.generate_embeddings(
        data=data,
        image_dir=path_to_images,
        out_dir=path_to_embeddings,
        model_name=embed_model_name
    )   
    embeddings.columns = [str(x) for x in embeddings.columns]
    n_features = len(embeddings.columns) - 1
    
    embeddings = embeddings[embeddings.UID.isin(data.UID.unique())]
    embeddings = embeddings.merge(data, on="UID", how="inner")    
    
    for target_column in target_columns:
        group_column = "grid_id"
        feature_columns = [str(x) for x in range(n_features)]

        X = embeddings[feature_columns]
        y = embeddings[target_column]
        groups = embeddings[group_column]

        cv, result, report, cm, preds = model_utils.group_kfold(config, X, y, groups)
        print(result)
        print(report)
        
        exp_name = f"{target_column.lower()}-{embed_model_name}-{model_name}-{args.iso_code}"
        exp_dir = os.path.join(cwd, f"exp/{args.iso_code}/{exp_name}")
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        preds.to_csv(os.path.join(exp_dir, f'{exp_name}.csv'))
        cm.plot().figure_.savefig(os.path.join(exp_dir, 'confusion_matrix.png'))
        
        with open(os.path.join(exp_dir, f'{exp_name}_result.json'), 'w') as file:
            json.dump(result, file)
        report.to_csv(os.path.join(exp_dir, f'{exp_name}_report.csv'))
        result["target"] = target_column
        
        run = wandb.init(project=config["project"], config=config)
        run.name = exp_name
        wandb_report = wandb.Table(dataframe=report)
        run.log(result)
        run.log({"report": wandb_report})
        run.finish()

        print(cv.best_estimator_)
        model = cv.best_estimator_.fit(X, y)
        joblib.dump(model, os.path.join(exp_dir, f'{exp_name}.joblib'))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--iso_code", help="ISO code", default="VCT")
    parser.add_argument("--mode", help="Aerial or streetview", default="aerial")
    parser.add_argument("--config", help="Main config", default="configs/config.yaml")
    parser.add_argument("--model_config", help="Model config", default="configs/model_configs/ResNet50_FMOW_RGB_GASSL-LR.yaml")
    args = parser.parse_args()
        
    main(args)
