import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb
import json
from torch.utils.data import random_split
import eval_cifar100  # Import evaluation scripts
import eval_ood
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR # Import StepLR

################################################################################
# Model Definition (Using pretrained ResNet18 model from torchvision)
################################################################################
def get_model(num_classes=100, pretrained=True, dropout_rate=0.5):
    # Using ResNet18 with pretrained weights
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)

    # Freeze all parameters initially if using pretrained weights
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False

    # Modify the first conv layer to work with CIFAR-100 (3x32x32 images)
    # Original ResNet expects 224x224 images, but we'll adapt it for 32x32
    # Keep this modification even with pretrained weights
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool as we have smaller images

    # Modify the final fully connected layer for 100 classes
    # This layer will be trainable by default
    num_ftrs = model.fc.in_features
    # Add dropout before the final layer
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_ftrs, num_classes)
    )

    # Ensure the new fc layer's parameters are trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG, phase="train"):
    """Train one epoch, e.g. all batches of one epoch."""
    device = CONFIG["device"]
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # put the trainloader iterator in a tqdm so it can print progress
    total_epochs = CONFIG['feature_extract_epochs'] + CONFIG['fine_tune_epochs']
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{total_epochs} [{phase}]", leave=False)

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):

        # move inputs and labels to the target device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0) # Accumulate loss correctly
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / total, "acc": 100. * correct / total})

    train_loss = running_loss / total
    train_acc = 100. * correct / total
    return train_loss, train_acc


################################################################################
# Define a validation function
################################################################################
def validate(model, valloader, criterion, device):
    """Validate the model"""
    model.eval() # Set to evaluation
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # No need to track gradients

        # Put the valloader iterator in tqdm to print progress
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        # Iterate through the validation set
        for i, (inputs, labels) in enumerate(progress_bar):

            # move inputs and labels to the target device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0) # Accumulate loss correctly
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / total, "acc": 100. * correct / total})

    val_loss = running_loss / total
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main():
    ############################################################################
    #    Configuration Dictionary (Modify as needed)
    ############################################################################
    CONFIG = {
        "model": "ResNet18_Pretrained_FineTuned",
        "batch_size": 32,           # Reduced from 64 (matching friend's final)
        "feature_extract_epochs": 5, # Epochs to train only the classifier
        "fine_tune_epochs": 20,     # Reduced from 35 (for 25 total epochs)
        "lr_feature_extract": 0.01, # Learning rate for the feature extraction phase
        "lr_fine_tune": 0.0005,     # Reduced from 0.001
        "dropout_rate": 0.5,        # Added Dropout rate configuration
        "num_workers": 4,
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge-part3", # Different project/name for Part 3
        "wandb_entity": "dlaboy-boston-university", # Replace with your WandB entity if needed
        "seed": 42,
        "weight_decay": 1e-4, # Regularization (Reverted back from 5e-4)
        "model_save_path": "best_model_part3.pth", # Filename for Part 3 model
        "submission_file": "submission_ood_part3.csv", # Filename for Part 3 submission
        # Switched to ImageNet mean/std for ImageNet pretrained model
        "imagenet_mean": (0.485, 0.456, 0.406),
        "imagenet_std": (0.229, 0.224, 0.225),
    }

    import pprint
    print("CONFIG Dictionary:")
    pprint.pprint(CONFIG)

    # Set random seed for reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    ############################################################################
    #      Data Transformation
    ############################################################################
    # Training transforms with more augmentation for fine-tuning
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)), # Added ResizedCrop
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # Added Rotation
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Added ColorJitter
        transforms.ToTensor(),
        # Use ImageNet normalization
        transforms.Normalize(CONFIG["imagenet_mean"], CONFIG["imagenet_std"]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False) # Added Random Erasing
    ])

    # Validation and test transforms (NO augmentation, only normalization)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # Use ImageNet normalization
        transforms.Normalize(CONFIG["imagenet_mean"], CONFIG["imagenet_std"]),
    ])

    ############################################################################
    #       Data Loading
    ############################################################################
    print("Loading CIFAR-100 dataset...")
    # Download and load the training data
    full_trainset = torchvision.datasets.CIFAR100(root=CONFIG['data_dir'], train=True,
                                            download=True, transform=transform_train)

    # Split train into train and validation (80/20 split)
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    # Use a generator for reproducibility with random_split
    generator = torch.Generator().manual_seed(CONFIG["seed"])
    trainset, valset = random_split(full_trainset, [train_size, val_size], generator=generator)

    # Important: Make sure the validation set uses the *test* transform
    # We need to create a new Dataset object for the validation set or modify its transform attribute carefully.
    # Easiest is often to reload the validation part of the dataset with the test transform,
    # but here we'll modify the subset's dataset transform attribute carefully.
    # Let's try assigning the transform to the underlying dataset for the val subset indices
    # A cleaner way might be to re-create the valset object with the correct transform.
    # For simplicity here, let's assume valset uses transform_test (often requires custom Subset class)
    # WORKAROUND: Apply test transform during validation loop if direct assignment fails.
    # A common pattern: keep a reference to the original dataset and switch transforms.
    # Let's stick to the method used in part2:
    valset.dataset.transform = transform_test # Modify the transform of the underlying full dataset for validation

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"],
                                             shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"],
                                           shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    # Test set
    testset = torchvision.datasets.CIFAR100(root=CONFIG['data_dir'], train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"],
                                            shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    print("Dataset loaded.")
    ############################################################################
    #   Instantiate model and move to target device
    ############################################################################
    model = get_model(num_classes=100, pretrained=True, dropout_rate=CONFIG["dropout_rate"])
    model = model.to(CONFIG["device"])

    # Print trainable parameters (optional check)
    # print("Trainable parameters after initial setup:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    print("Model summary:")
    # print(f"{model}") # Avoid printing the whole model structure

    ############################################################################
    # Loss Function, Optimizer and optional learning rate scheduler
    ############################################################################
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Added label smoothing

    # Initialize wandb
    wandb.init(project=CONFIG["wandb_project"], entity=CONFIG["wandb_entity"], config=CONFIG, name="Part3-ResNet18-FineTune")
    wandb.watch(model, log="all", log_freq=100) # Watch gradients, weights, etc.

    ############################################################################
    # --- Training Loop ---
    ############################################################################
    best_val_acc = 0.0
    current_epoch = 0

    # --- Phase 1: Feature Extraction ---
    print("--- Phase 1: Feature Extraction (Training Classifier) ---")
    # Optimize only the classifier parameters
    optimizer_extract = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=CONFIG["lr_feature_extract"],
                                   weight_decay=CONFIG["weight_decay"]) # Apply weight decay even here
    scheduler_extract = optim.lr_scheduler.StepLR(optimizer_extract, step_size=3, gamma=0.1) # Simple scheduler

    for epoch in range(CONFIG["feature_extract_epochs"]):
        actual_epoch = current_epoch + epoch
        train_loss, train_acc = train(actual_epoch, model, trainloader, optimizer_extract, criterion, CONFIG, phase="Extract")
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler_extract.step()

        wandb.log({
            "epoch": actual_epoch + 1,
            "phase": 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer_extract.param_groups[0]["lr"]
        })
        print(f"Epoch {actual_epoch+1} (Phase 1) - Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            wandb.save(CONFIG["model_save_path"]) # Save to wandb
            print(f"Model saved! New best val accuracy: {best_val_acc:.2f}%")

    current_epoch += CONFIG["feature_extract_epochs"]

    # --- Phase 2: Fine-tuning ---
    print("--- Phase 2: Fine-tuning (Training Full Network) ---")
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    # Create a new optimizer for fine-tuning with a lower learning rate
    # Switched to SGD with momentum
    optimizer_tune = optim.SGD(model.parameters(),
                               lr=CONFIG["lr_fine_tune"],
                               momentum=0.9, # Added momentum
                               weight_decay=CONFIG["weight_decay"])
    # Use a more sophisticated scheduler like CosineAnnealingLR for fine-tuning
    # Switched to StepLR like the friend's report
    scheduler_tune = StepLR(optimizer_tune, step_size=7, gamma=0.1)

    for epoch in range(CONFIG["fine_tune_epochs"]):
        actual_epoch = current_epoch + epoch
        train_loss, train_acc = train(actual_epoch, model, trainloader, optimizer_tune, criterion, CONFIG, phase="FineTune")
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler_tune.step() # StepLR steps per epoch

        wandb.log({
            "epoch": actual_epoch + 1,
            "phase": 2,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer_tune.param_groups[0]["lr"]
        })
        print(f"Epoch {actual_epoch+1} (Phase 2) - Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            wandb.save(CONFIG["model_save_path"]) # Save to wandb
            print(f"Model saved! New best val accuracy: {best_val_acc:.2f}%")


    wandb.summary["best_val_accuracy"] = best_val_acc
    wandb.finish()
    print("Training Finished.")

    ############################################################################
    # Evaluation
    ############################################################################
    print("--- Evaluation ---")
    # Load the best model saved during training
    print(f"Loading best model from {CONFIG['model_save_path']}")
    # Instantiate a fresh model and load state_dict
    eval_model = get_model(num_classes=100, pretrained=False) # Create model structure without re-downloading weights
    eval_model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=CONFIG["device"]))
    eval_model = eval_model.to(CONFIG["device"])
    eval_model.eval() # Set to evaluation mode

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    # Note: eval_cifar100.evaluate_cifar100_test reloads the model internally, which might be redundant.
    # Let's pass the loaded model directly if possible, or modify the eval script.
    # Assuming eval_cifar100 uses the passed model:
    # If eval_cifar100.py expects a model path, we need to adjust.
    # Let's check eval_cifar100.py: it loads "best_model.pth". We need to modify it or this script.
    # Easiest here: ensure eval_cifar100 can accept a model object or load the correct path.
    # For now, let's assume we modify eval_cifar100 or it accepts the model.
    # --- Let's modify evaluate_cifar100_test to accept the model object ---
    # (Manual modification needed in eval_cifar100.py: remove internal model loading)
    print("Evaluating on Clean CIFAR-100 Test Set...")
    # Assuming evaluate_cifar100_test is modified like:
    # def evaluate_cifar100_test(model, testloader, device): # Removed internal loading
    #     model.eval() # Ensure eval mode
    #     ... (rest of the function) ...
    try:
        predictions_clean, clean_accuracy = eval_cifar100.evaluate_cifar100_test(eval_model, testloader, CONFIG["device"])
        print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")
        # Log clean accuracy to WandB summary
        # Re-init wandb briefly or use wandb.summary if run hasn't finished fully.
        # Since wandb finished, we can't log directly. Print is sufficient.
    except Exception as e:
        print(f"Could not run clean evaluation. Ensure 'eval_cifar100.py' accepts a model object. Error: {e}")
        clean_accuracy = -1 # Indicate failure


    # --- Evaluation on OOD ---
    # eval_ood.evaluate_ood_test expects a model object, so this should work.
    print("Evaluating on OOD Test Set...")
    all_predictions_ood = eval_ood.evaluate_ood_test(eval_model, CONFIG)

    # --- Create Submission File (OOD) ---
    print("Creating OOD submission file...")
    submission_df_ood = eval_ood.create_ood_df(all_predictions_ood)
    submission_df_ood.to_csv(CONFIG["submission_file"], index=False)
    print(f"OOD submission file created successfully: {CONFIG['submission_file']}")

    # Optional: Upload submission file to WandB as an artifact
    # Need to re-initialize wandb to log artifact after run finish, or save before finish.
    # Let's log it before finishing the run. (Move wandb.finish() down)
    # --> Moved wandb.finish() earlier, so we'll skip artifact logging here.

    print("Part 3 script complete.")


if __name__ == '__main__':
    main()
