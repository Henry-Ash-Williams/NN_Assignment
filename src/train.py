# import the os module for file based operations 
import os
# import the command line argument parser library 
import argparse
# import various type annotations 
from typing import Tuple, List, Literal, Optional

# Torch stuff
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim import Optimizer
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
# Numpy 
import numpy as np
# Sklearn 
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb

import model 

# Read arguments from the command line to make automated
# testing easier
argparser = argparse.ArgumentParser()
argparser.add_argument("--learning-rate", type=float, default=2e-5)
argparser.add_argument("--batch-size", type=int, default=32)
argparser.add_argument("--dropout-rate", type=float, default=0.1)
argparser.add_argument("--weight-decay", type=float, default=0.01)
argparser.add_argument("--epochs", type=int, default=10)
argparser.add_argument("--optimizer", type=str, default="AdamW")
argparser.add_argument("--experiment-id", type=int, default=1)
argparser.add_argument("--experiment-name", type=str, default="baseline")
argparser.add_argument("--experiment-description", type=str, default="No Description Provided")
argparser.add_argument("--run-id", type=int)
argparser.add_argument("--model-name", type=str, default="ImageClassifier")
argparser.add_argument("--save", choices=["model", "cm", "both", "none"], type=str, default='none')
argparser.add_argument("--freeze-conv-layers", type=bool, default=False)
argparser.add_argument("--run-name", type=str)
argparser.add_argument("--tag", type=str)
args = argparser.parse_args()

# Prefer mps over cuda because I use a mac
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# Directory to save model & classification metrics
OUT_DIR = os.path.join(
    os.getcwd(), f"models/experiment-{args.experiment_id}/{args.experiment_name}"
) if args.save in ['model', 'cm', 'both'] else None

# Create output directory if the user wants to save stuff 
if args.save in ['model', 'cm', 'both'] and not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Initial data transform
# convert the image to a tensor, and then normalize the pixel values from [0, 1] 
# to [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
# Loss function 
criterion = nn.CrossEntropyLoss()

# def scheduler()


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """
    Plot the confusion matrix and save it to disk

    Args:
        y_true (List): True labels.
        y_pred (List): Predicted labels.
        labels (List[str]): List of class labels.
    """
    plt.ioff()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure()
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)


def evaluate(
    model: nn.Module, dataset: DataLoader
) -> Tuple[List[np.int64], List[np.int64], float]:
    """
    Evaluates a neural network model on a given dataset.

    Args:
        model (nn.Module): The neural network model to evaluate.
        dataset (DataLoader): The DataLoader providing the dataset for evaluation.

    Returns:
        Tuple[List[np.int64], List[np.int64], float]: A tuple containing:
            - List of predicted labels (List[np.int64])
            - List of true labels (List[np.int64])
            - Total loss (float) accumulated over the dataset

    The function sets the model to evaluation mode, iterates over the dataset, and computes
    the predictions and the loss for each batch without updating the model parameters. It
    accumulates the total loss and collects the predicted and true labels for each sample.
    The predicted labels are determined by taking the argmax of the model's output for each
    sample.

    Note:
        This function assumes that a global variable `DEVICE` is defined, indicating the device
        (CPU or GPU) on which the computation should be performed, and a global `criterion` is
        defined for calculating the loss.
    """
    model.eval()
    total_loss = 0
    true_labels = []
    predicted_labels = []

    for images, labels in dataset:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            output = model(images)
            loss = criterion(output, labels)

        predictions = output.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        loss = loss.item()
        total_loss += loss
        predicted_labels.append(predictions)
        true_labels.append(labels)

    predicted_labels = [
        np.argmax(prediction)
        for batched_predictions in predicted_labels
        for prediction in batched_predictions
    ]
    true_labels = [label for batched_labels in true_labels for label in batched_labels]
    return predicted_labels, true_labels, total_loss


def get_datasets() -> Tuple[DataLoader, DataLoader]:
    train, val = random_split(
        torchvision.datasets.CIFAR10(
            root="./cifar10", train=True, download=True, transform=transform
        ),
        [0.5, 0.5],
    )

    
    test = torchvision.datasets.CIFAR10(
        root="./cifar10", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val, shuffle=False, batch_size=args.batch_size)
    test_loader = DataLoader(test, shuffle=False, batch_size=args.batch_size)

    return train_loader, val_loader, test_loader


def get_optimizer(optimizer_name: str, model: nn.Module) -> Optimizer:
    """
    Get an instance of the specified optimizer.

    Args:
        optimizer_name (str): The name of the optimizer class.
        model: (torch.nn.Module): The model to perform optmization on

    Returns:
        torch.optim.Optimizer: An instance of the specified optimizer class.

    Example:
        optimizer = get_optimizer('Adam')
    """
    try:
        optimizer_cls = getattr(optim, optimizer_name)
    except AttributeError:
        print(f"Could not find a pytorch optimizer with the name {optimizer_name}")

    if args.freeze_conv_layers: 
        for conv_layer in [model.conv1, model.conv2, model.conv3]:
            for parameter in conv_layer:
                parameter.requires_grad = False
        return optimizer_cls(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    else: 
        return optimizer_cls(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


def get_model(model_name: str) -> nn.Module:
    """
    Get an instance of the model.

    Args:
        model_name (str): The name of the model.

    Returns:
        torch.nn.Model: An instance of the specified model.

    Example:
        model = get_model("ImageClassifier")
    """
    if model_name == "ImageClassifier":
        return getattr(model, model_name)()
    elif model_name == "ImageClassifierWithDropout":
        return getattr(model, model_name)(args.dropout_rate)
    else:
        raise ValueError(f"Could not find model with the name {model_name}")


def train(
    save_model: bool=False, save_confusion_matrix: bool=False
) -> Tuple[List[np.int64], List[np.int64], float]:
    """
    Trains a neural network model and optionally saves the model and confusion matrix.

    Args:
        save_model (bool, optional): Whether to save the trained model. Defaults to False.
        save_confusion_matrix (bool, optional): Whether to save the confusion matrix plot. Defaults to False.

    Returns:
        Tuple[List[np.int64], List[np.int64], float]: A tuple containing:
            - List of predicted labels from the test dataset (List[np.int64])
            - List of true labels from the test dataset (List[np.int64])
            - Total loss (float) on the test dataset

    The function sets up the model, optimizer, and data loaders, and then trains the model
    for a specified number of epochs. During each epoch, it logs the training loss and
    evaluates the model on the test dataset, logging various evaluation metrics. If specified,
    it saves the trained model and/or the confusion matrix to the output directory.

    Note:
        This function assumes the presence of global variables and functions such as `args`,
        `DEVICE`, `criterion`, `wandb`, `get_model`, `get_optimizer`, `get_datasets`, 
        `OUT_DIR`, and `plot_confusion_matrix`. It also assumes the datasets are set up for 
        training and testing.

    Example:
        metrics = train(save_model=True, save_confusion_matrix=True)
    """
    # Gets an instance of model based on the argument '--model-name' argument 
    # and send it to the gpu 
    image_classifier = get_model(args.model_name).to(DEVICE)

    # Gets an instance of optimzer based on the argument '--optimizer' argument 
    optimizer = get_optimizer(args.optimizer, image_classifier)
    # Get the datasets 
    train_loader, val_loader, test_loader = get_datasets()
    
    # cache the number of samples within each of the datasets 
    n_samples = {"train": len(train_loader.dataset), "test": len(test_loader.dataset), "val": len(val_loader.dataset), }

    # Loop for the number of epochs specified '--epochs' 
    for epoch in range(args.epochs):
        # Set the model to training mode 
        image_classifier.train()
        # Record the total training loss each epoch 
        total_training_loss = 0
        # Pretty display bar uwu 
        train_iter = tqdm(
            train_loader, desc=f"Training epoch {epoch + 1:02}/{args.epochs:02}"
        )

        # Go over each batch in the dataset 
        for images, labels in train_iter:
            # Send the batches and labels to the GPU 
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Reset the gradients of all optimzed tensors 
            optimizer.zero_grad()
            # Perform a forward pass on the model with the training data 
            output = image_classifier.forward(images)
            # Calculate the loss 
            loss = criterion(output, labels)
            # Backpropagate 
            loss.backward()
            optimizer.step()
            # Add loss for this batch to the total loss 
            total_training_loss += loss.item()
            # Log this batches loss to weights and biases 
            wandb.log({"train": {"loss": loss.item()}})

        # Perform validation 
        val_predictions, val_labels, total_val_loss = evaluate(
            image_classifier, test_loader
        )

        # Log average loss for this epoch 
        wandb.log({"train": {"avg_loss": total_training_loss / n_samples["train"]}})

        # Calculate validation metrics
        metrics = classification_report(
            val_labels, val_predictions, output_dict=True, zero_division=0.0
        )
        # Log the weighted average of the validation metrics 
        wandb.log(
            {
                "val": {
                    "accuracy": metrics["accuracy"],
                    "f1_score": metrics["weighted avg"]["f1-score"],
                    "precision": metrics["weighted avg"]["precision"],
                    "recall": metrics["weighted avg"]["recall"],
                    "avg_loss": (total_val_loss / n_samples["test"]),
                }
            }
        )
    # Evaluate the model on the test set 
    metrics = evaluate(image_classifier, test_loader)

    # If the user wants to save the model or the confusion matrix, 
    # determine where they should be saved 
    if args.save in ['model', 'cm', 'both']: 
        num_runs = len(os.listdir(OUT_DIR))
        save_dir = os.path.join(OUT_DIR, f"{num_runs:04}")

    # create the save directory 
    if save_model or save_confusion_matrix:
        os.makedirs(save_dir)

    # Save the model 
    if save_model:
        model_path = os.path.join(save_dir, "model.pt")
        torch.save(image_classifier.state_dict(), model_path)
        model_artifact = wandb.Artifact(name="image-classification-model", type="model")
        model_artifact.add_file(local_path=model_path)
        wandb.log_artifact(model_artifact)

    # Save the confusion matrix 
    if save_confusion_matrix:
        plot_path = os.path.join(save_dir, "confusion_matrix.png")
        plot_confusion_matrix(metrics[1], metrics[0], list(range(10)), plot_path)
        image = wandb.Image(plot_path, mode="RGB")
        wandb.log({"Confusion Matrix": image})

    # Return the testing metrics 
    return metrics


if __name__ == "__main__":
    wandb.login() # Log into weights an biases
    # Tells weights and biases to use a certain project, what hyperparameters to use
    # information about the run, and any tags to associate with it 
    wandb.init(
        project="Neural Networks Assessment",
        notes=args.experiment_description,
        config={
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "dropout_rate": args.dropout_rate,
            "decay_rate": args.weight_decay,
            "optimizer": args.optimizer,
        },
        name=args.run_name,
        tags=None if args.tag is None else [args.tag]
    )
    # Train the model 
    predicted, true_labels, loss = train(save_model=(args.save in ['model', 'both']), save_confusion_matrix=(args.save in ['cm', 'both']))

    # Create a classifcation report of the results of the test set 
    metrics = classification_report(
        predicted, true_labels, output_dict=True, zero_division=0.0
    )

    # Include the metrics from the testing metrics in the summary of this run 
    wandb.run.summary["accuracy"] = metrics["accuracy"]
    wandb.run.summary["precision"] = metrics["weighted avg"]["precision"]
    wandb.run.summary["recall"] = metrics["weighted avg"]["recall"]
    wandb.run.summary["f1"] = metrics["weighted avg"]["f1-score"]

    # Shutdown weights and biases
    wandb.finish()
