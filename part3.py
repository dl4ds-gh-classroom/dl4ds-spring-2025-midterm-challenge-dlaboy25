import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import wandb
from torch.utils.data import random_split

################################################################################
# Model Definition (Transfer Learning with Pretrained ResNet-18)
################################################################################
def get_model(num_classes=100):
    # Load pretrained ResNet-18
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    
    # Modify the first conv layer to work with CIFAR-100 (3x32x32 images)
    # Original ResNet has 7x7 conv with stride 2, we'll use 3x3 with stride 1 for smaller images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.bn1 = nn.BatchNorm2d(64)
    
    # Remove the maxpool layer as we have smaller images
    model.maxpool = nn.Identity()
    
    # Modify the final fully connected layer for 100 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Add dropout for regularization
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model

################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch, e.g. all batches of one epoch."""
    device = CONFIG["device"]
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # put the trainloader iterator in a tqdm so it can print progress
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

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

        running_loss += loss.item() * inputs.size(0)
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

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / total, "acc": 100. * correct / total})

    val_loss = running_loss / total
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main():
    ############################################################################
    # Configuration Dictionary
    ############################################################################
    CONFIG = {
        "model": "ResNet18-Pretrained",
        "batch_size": 128,
        "learning_rate": 0.001,  # Lower initial LR for fine-tuning
        "epochs": 40,
        "num_workers": 4,
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
        "weight_decay": 1e-4,  # Regularization
        "momentum": 0.9,  # For SGD optimizer
        # Training strategy
        "phase1_epochs": 5,  # Frozen backbone phase
        "phase2_epochs": 35, # Full fine-tuning phase
        # CIFAR-100 stats
        "cifar100_mean": (0.5071, 0.4867, 0.4408),
        "cifar100_std": (0.2675, 0.2565, 0.2761),
    }

    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    # Set random seed for reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    ############################################################################
    # Data Transformation with Advanced Augmentation
    ############################################################################
    # Training transforms with augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # Rotation augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
        transforms.ToTensor(),
        transforms.Normalize(CONFIG["cifar100_mean"], CONFIG["cifar100_std"]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # Random erasing augmentation
    ])

    # Validation and test transforms (NO augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CONFIG["cifar100_mean"], CONFIG["cifar100_std"]),
    ])

    ############################################################################
    # Data Loading
    ############################################################################
    # Download and load the training data
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                           download=True, transform=transform_train)

    # Split train into train and validation (90/10 split) - using more data for training
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])
    
    # Update valset transform to use test transform (no augmentation)
    valset.dataset.transform = transform_test
    
    # Create data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"],
                                             shuffle=True, num_workers=CONFIG["num_workers"],
                                             pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"],
                                           shuffle=False, num_workers=CONFIG["num_workers"],
                                           pin_memory=True)

    # Test set
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"],
                                            shuffle=False, num_workers=CONFIG["num_workers"],
                                            pin_memory=True)
    
    ############################################################################
    # Instantiate model and move to target device
    ############################################################################
    model = get_model(num_classes=100)  # CIFAR-100 has 100 classes
    model = model.to(CONFIG["device"])

    print("\nModel summary:")
    print(f"{model}\n")

    ############################################################################
    # Training Strategy: 2-Phase Fine-tuning
    ############################################################################
    print("Starting 2-phase fine-tuning strategy...")
    
    # Phase 1: Freeze backbone, train only final layer
    print("Phase 1: Freeze backbone, train only final layer...")
    
    # Freeze all parameters except the final layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Phase 1 criterion, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    # Use Adam for the first phase as it's more suitable for training only a few parameters
    optimizer_phase1 = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=CONFIG["weight_decay"])
    scheduler_phase1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase1, T_max=CONFIG["phase1_epochs"])

    # Initialize wandb
    run_name = f"{CONFIG['model']}-phase1"
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG, name=run_name)
    wandb.watch(model)

    # Phase 1 Training Loop
    best_val_acc = 0.0
    
    for epoch in range(CONFIG["phase1_epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer_phase1, criterion, 
                                      {**CONFIG, "epochs": CONFIG["phase1_epochs"]})
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler_phase1.step()
        
        # Log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer_phase1.param_groups[0]["lr"],
            "phase": 1
        })
        
        print(f"Phase 1 - Epoch {epoch+1}/{CONFIG['phase1_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_phase1.pth")
            print(f"Phase 1 - New best val accuracy: {best_val_acc:.2f}%")
    
    wandb.finish()
    
    # Phase 2: Unfreeze everything and fine-tune the entire model
    print("\nPhase 2: Fine-tune the entire model...")
    
    # Load the best model from Phase 1
    model.load_state_dict(torch.load("best_model_phase1.pth"))
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Phase 2 criterion, optimizer and scheduler
    # Use SGD with momentum for the second phase to fine-tune the entire model
    optimizer_phase2 = optim.SGD(model.parameters(), lr=0.0005, 
                                momentum=CONFIG["momentum"], 
                                weight_decay=CONFIG["weight_decay"])
                                
    # One-cycle learning rate scheduler for faster convergence
    scheduler_phase2 = optim.lr_scheduler.OneCycleLR(
        optimizer_phase2, 
        max_lr=0.01,  # Maximum learning rate
        epochs=CONFIG["phase2_epochs"],
        steps_per_epoch=len(trainloader),
        pct_start=0.2,  # Percentage of iterations for increasing LR
        anneal_strategy='cos',  # Cosine annealing
        div_factor=25,   # Initial learning rate is max_lr/div_factor
        final_div_factor=10000,  # Final learning rate is max_lr/(div_factor*final_div_factor)
    )
    
    # Initialize wandb for phase 2
    run_name = f"{CONFIG['model']}-phase2"
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG, name=run_name)
    wandb.watch(model)
    
    # Phase 2 Training Loop
    best_val_acc = 0.0
    for epoch in range(CONFIG["phase2_epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer_phase2, criterion, 
                                      {**CONFIG, "epochs": CONFIG["phase2_epochs"]})
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        
        # Log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer_phase2.param_groups[0]["lr"],
            "phase": 2
        })
        
        print(f"Phase 2 - Epoch {epoch+1}/{CONFIG['phase2_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Phase 2 - New best val accuracy: {best_val_acc:.2f}%")
        
        # Update learning rate after each batch
        scheduler_phase2.step()
        
    wandb.finish()

    ############################################################################
    # Evaluation
    ############################################################################
    print("\nRunning final evaluation...")
    import eval_cifar100
    import eval_ood

    # Load the best model
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    print("Evaluating on out-of-distribution data...")
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood_part3.csv", index=False)
    print("submission_ood_part3.csv created successfully.")

if __name__ == '__main__':
    main()
