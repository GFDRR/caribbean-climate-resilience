import os
import shutil
import time
import argparse
import logging
import geopandas as gpd
import pandas as pd
import yaml

import torch
import wandb
import json

import sys

sys.path.insert(0, "./utils/")
import cv_utils
import eval_utils

torch.cuda.empty_cache()

# Get device
cwd = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")


def main(args):
    # Get current working directory
    cwd = os.getcwd()

    # Load main config and model-specific config, and merge them
    with open(args.config, "r") as stream:
        c = yaml.safe_load(stream)
    with open(args.model_config, "r") as stream:
        c.update(yaml.safe_load(stream))

    logging.info(c)

    # Add additional parameters from command-line arguments
    c["iso_code"] = args.iso_code
    c["target"] = args.target
    c["mode"] = args.mode

    c["path_to_images"] = c["path_to_images"].format(
        mode=args.mode, iso_code=args.iso_code
    )
    c["path_to_file"] = c["path_to_file"].format(mode=args.mode, iso_code=args.iso_code)

    # Set wandb configs
    wandb.init(project=c["project"], config=c)

    # Create experiment folder
    exp_name = f"{c['target']}-{c['model']}-{c['iso_code']}"
    exp_dir = os.path.join(cwd, "exp/", c["iso_code"], exp_name)
    c["exp_dir"] = exp_dir
    logging.info(f"Experiment directory: {exp_dir}")
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir)

    # Initialize logging
    logname = os.path.join(exp_dir, f"{exp_name}.log")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    handler = logging.FileHandler(logname)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logging.info(exp_name)
    wandb.run.name = exp_name

    # Load dataset
    phases = ["train", "test"]
    data, data_loader, classes = cv_utils.load_dataset(config=c, phases=phases)
    logging.info(classes)
    logging.info(f"Train/val/test sizes: {len(data['train'])}/{len(data['test'])}")
    wandb.log({f"{phase}_size": len(data[phase]) for phase in phases})

    # Load model, optimizer, and scheduler
    model, criterion, optimizer, scheduler = cv_utils.load_model(
        n_classes=len(classes),
        model_type=c["model"],
        pretrained=c["pretrained"],
        scheduler_type=c["scheduler"],
        optimizer_type=c["optimizer"],
        label_smoothing=c["label_smoothing"],
        lr=c["lr"],
        patience=c["patience"],
        data_loader=data_loader,
        device=device,
    )
    logging.info(model)

    lr = optimizer.param_groups[0]["lr"]
    logging.info(f"LR: {lr}")
    wandb.log({"lr": lr})

    # Instantiate wandb tracker
    wandb.watch(model)

    # Commence model training
    n_epochs = c["n_epochs"]
    scorer = c["scorer"]
    since = time.time()
    best_score = -1
    best_results = None

    for epoch in range(1, n_epochs + 1):
        logging.info("\nEpoch {}/{}".format(epoch, n_epochs))

        # Train model
        cv_utils.train(
            data_loader["train"],
            model,
            criterion,
            optimizer,
            device,
            wandb=wandb,
            logging=logging,
        )
        # Evauate model
        phase = "test"
        val_results, val_report, val_cm, val_preds = cv_utils.evaluate(
            data_loader[phase],
            class_names=classes,
            model=model,
            criterion=criterion,
            device=device,
            phase=phase,
            wandb=wandb,
            logging=logging,
        )
        scheduler.step(val_results[f"{phase}_loss"])

        # Save best model so far
        if val_results[f"{phase}_{scorer}"] > best_score:
            best_score = val_results[f"{phase}_{scorer}"]
            best_results = val_results

            model_file = os.path.join(exp_dir, f"{exp_name}.pth")
            torch.save(model.state_dict(), model_file)

            val_preds.to_csv(os.path.join(c["exp_dir"], f"{exp_name}.csv"))
            val_cm.plot().figure_.savefig(
                os.path.join(c["exp_dir"], "confusion_matrix.png")
            )
            with open(
                os.path.join(c["exp_dir"], f"{exp_name}_result.json"), "w"
            ) as file:
                json.dump(val_results, file)
            val_report.to_csv(os.path.join(c["exp_dir"], f"{exp_name}_report.csv"))

        logging.info(f"Best {phase}_{scorer}: {best_score}")
        log_results = {key: val for key, val in best_results.items() if key[-1] != "_"}
        logging.info(f"Best scores: {log_results}")

        # Terminate if learning rate becomes too low
        learning_rate = optimizer.param_groups[0]["lr"]
        if learning_rate < c["lr_min"]:
            break

    # Terminate trackers
    time_elapsed = time.time() - since
    logging.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    # Load best model
    model_file = os.path.join(exp_dir, f"{exp_name}.pth")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)

    # Calculate final test performance using best model
    phase = "test"
    logging.info(f"\n{phase.capitalize()} Results")
    test_results, test_report, test_cm, test_preds = cv_utils.evaluate(
        data_loader[phase],
        class_names=classes,
        model=model,
        criterion=criterion,
        device=device,
        phase=phase,
        wandb=wandb,
        logging=logging,
    )
    test_preds.to_csv(os.path.join(c["exp_dir"], f"{exp_name}.csv"))
    test_cm.plot().figure_.savefig(os.path.join(c["exp_dir"], "confusion_matrix.png"))
    with open(os.path.join(c["exp_dir"], f"{exp_name}_result.json"), "w") as file:
        json.dump(test_results, file)
    test_report.to_csv(os.path.join(c["exp_dir"], f"{exp_name}_report.csv"))
    return test_results


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--iso_code", help="ISO code", default="VCT")
    parser.add_argument("--target", help="Target class", default="roof_material")
    parser.add_argument("--mode", help="Aerial or streetview", default="aerial")
    parser.add_argument(
        "--config", help="Path to the configuration file", default="configs/config.yaml"
    )
    parser.add_argument(
        "--model_config",
        help="Model config",
        default="configs/cv_configs/convnext_base.yaml",
    )
    args = parser.parse_args()

    main(args)
