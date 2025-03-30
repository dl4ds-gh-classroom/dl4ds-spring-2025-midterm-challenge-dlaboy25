import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
from tqdm.auto import tqdm
import wandb


from eval_ood import evaluate_ood_test, create_ood_df

CONFIG = {
    "epochs": 10, 
    "batch_size": 128,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "data_dir": "./data",
    "ood_dir": "./data/ood-test",
    "model_save_path": "./simple_cnn_best.pth",
    "submission_file": "./submission_part1.csv",
    "wandb_project": "dl4ds_midterm_part1",
    "wandb_entity": "dlaboy-boston-university", 
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # CIFAR-100 mean/std (calculated from the training set)
    # These are standard values, similar to those in eval_ood.py
    "cifar100_mean": (0.5071, 0.4867, 0.4408),
    "cifar100_std": (0.2675, 0.2565, 0.2761),
}

os.makedirs(CONFIG["ood_dir"], exist_ok=True)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(SimpleCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)  # 32x32x32
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)  # 16x16x32
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 16x16x64
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)  # 8x8x64
        
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, padding=1)  # 8x8x96
        self.bn3 = nn.BatchNorm2d(96)
        self.pool3 = nn.MaxPool2d(2)  # 4x4x96
        
        # Classifier part
        self.fc1 = nn.Linear(96 * 4 * 4, 192)
        self.dropout = nn.Dropout(0.4)  
        self.fc2 = nn.Linear(192, num_classes)
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 96 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# --- Data Loading --- #
def load_cifar100(config):
    """Loads CIFAR-100 dataset with specified transforms."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # Augmentation
        transforms.RandomHorizontalFlip(),          # Augmentation
        transforms.ToTensor(),
        transforms.Normalize(config["cifar100_mean"], config["cifar100_std"]),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config["cifar100_mean"], config["cifar100_std"]),
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=config["data_dir"],
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = torchvision.datasets.CIFAR100(
        root=config["data_dir"],
        train=False,
        download=True,
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    return train_loader, val_loader

# --- Training Function ---
def train(model, train_loader, criterion, optimizer, device):
    """Performs one epoch of training."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar description
        progress_bar.set_postfix({
            'loss': running_loss / total,
            'acc': 100. * correct / total
        })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# --- Evaluation Function ---
def evaluate(model, val_loader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar description
            progress_bar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# --- Main Execution Function --- #
def main():
    # Initialize WandB
    # Use the globally defined CONFIG dictionary
    wandb.init(project=CONFIG["wandb_project"], entity=CONFIG["wandb_entity"], config=CONFIG)

    print("Using device:", CONFIG["device"])
    device = torch.device(CONFIG["device"])

    # Load Data
    print("Loading CIFAR-100 dataset...")
    train_loader, val_loader = load_cifar100(CONFIG)
    print("Dataset loaded.")

    # Initialize Model, Loss, Optimizer
    model = SimpleCNN(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    # Optional: Add a learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # Log model architecture to WandB (optional)
    wandb.watch(model, log="all")

    # Training Loop
    best_val_acc = 0.0
    print("Starting training...")
    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        # --- Ensure correct calls inside main() ---
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        # --- End Ensure ---

        # Step the scheduler based on validation accuracy
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr'] # Log current learning rate
        })

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"Saving best model with Val Acc: {best_val_acc:.2f}% to {CONFIG['model_save_path']}")
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            wandb.summary["best_val_accuracy"] = best_val_acc # Update summary in WandB

    print("\nTraining finished.")

    # Load the best model for OOD evaluation
    print(f"\nLoading best model from {CONFIG['model_save_path']} for OOD evaluation...")
    # Ensure model is initialized before loading state_dict
    best_model = SimpleCNN(num_classes=100).to(device)
    best_model.load_state_dict(torch.load(CONFIG['model_save_path'], map_location=device))
    best_model.eval() # Set to evaluation mode

    # Perform OOD Evaluation
    print("Starting OOD evaluation...")
    # Pass the whole CONFIG dictionary to the OOD functions
    # Ensure the normalization used here matches the OOD evaluation expectations
    ood_predictions = evaluate_ood_test(best_model, CONFIG)
    print("OOD evaluation finished.")

    # Create Submission File
    print("Creating submission file...")
    submission_df = create_ood_df(ood_predictions)
    submission_df.to_csv(CONFIG["submission_file"], index=False)
    print(f"Submission file saved to {CONFIG['submission_file']}")

    # Optional: Upload submission file to WandB as an artifact
    artifact = wandb.Artifact('submission-part1', type='submission')
    artifact.add_file(CONFIG["submission_file"])
    wandb.log_artifact(artifact)

    # Finish WandB run
    wandb.finish()
    print("WandB run finished.")

    print("\nPart 1 script complete.")

# --- Call the main function --- #
if __name__ == "__main__":
    main() # Call the main execution function