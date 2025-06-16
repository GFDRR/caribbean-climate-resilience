import pandas as pd
from sklearn.metrics import (
    make_scorer,
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


def get_confusion_matrix(y_test, y_pred, class_names, encoded=False):
    """Generates the confusion matrix given the predictions
    and ground truth values.

    Args:
        y_test (list or numpy array): A list of ground truth values.
        y_pred (list of numpy array): A list of prediction values.
        class_names (list): A list of string labels or class names.

    Returns:
        pandas DataFrame: The confusion matrix.
        pandas DataFrame: A dataframe containing the precision,
            recall, and F1 score per class.
    """

    y_pred = pd.Series(y_pred, name="Predicted")
    y_test = pd.Series(y_test, name="Actual")

    labels = class_names if not encoded else [x for x in range(len(class_names))]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    cm_metrics = _get_cm_metrics(cm, list(cm.columns))
    return cm, cm_display, cm_metrics


def _get_cm_metrics(cm, class_names):
    """Return the precision, recall, and F1 score per class.

    Args:
        cm (pandas DataFrame or numpy array): The confusion matrix.
        class_names (list): A list of string labels or class names.

    Returns:
        pandas DataFrame: A dataframe containing the precision,
            recall, and F1 score per class.
    """

    metrics = {}
    for i in class_names:
        tp = cm.loc[i, i]
        fn = cm.loc[i, :].drop(i).sum()
        fp = cm.loc[:, i].drop(i).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 / (recall**-1 + prec**-1) if prec + recall > 0 else 0

        scores = {"precision": prec * 100, "recall": recall * 100, "f1_score": f1 * 100}

        metrics[i] = scores
    metrics = pd.DataFrame(metrics).T

    return metrics


def evaluate(y_true, y_pred):
    """Returns a dictionary of performance metrics.

    Args:
        y_test (list or numpy array): A list of ground truth values.
        y_pred (list of numpy array): A list of prediction values.

    Returns:
        dict: A dictionary of performance metrics.
    """
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    report = pd.DataFrame(report).transpose()

    results = {
        "overall_accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "recall_score": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_score": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    return results, report


def get_scoring():
    """Returns the dictionary of scorer objects."""

    return {
        "overall_accuracy": make_scorer(accuracy_score),
        "kappa": make_scorer(cohen_kappa_score),
        "recall_score": make_scorer(recall_score, average="macro", zero_division=0),
        "precision_score": make_scorer(
            precision_score, average="macro", zero_division=0
        ),
        "f1_score": make_scorer(f1_score, average="macro", zero_division=0),
    }
