
from model_3d import Gesture3DNet
from stack_3d_dataset import Stacked3DDataset
from itertools import product
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.cuda import is_available
import numpy as np
import random
import tqdm
import os

device = "cuda" if is_available() else "cpu"
torch.cuda.empty_cache()

SEED = 30
MODEL_PATH = "models/model.pt"
DATA_DIR = "rec_dataset"
ANNOTATIONS_DIR = "rec_dataset/data.csv"
LOG_VAL = "rec_dataset/cache_dir/stack_val_results.txt"
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
MAX_FLOWS = 60
EPOCHS = 100
BATCH_SIZE = 8

CONFIG = {
    "image_width" : IMAGE_WIDTH,
    "image_height" : IMAGE_HEIGHT,
    "magnitude_thresh" : 1,
    "max_flows" : MAX_FLOWS,
    "farneback_params" : {
        "pyr_scale": 0.5,
        "levels": 4,
        "winsize": 21,
        "iterations": 5,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }
}

def generate_configs():
    search_space = {
    "magnitude_thresh": [1, 2.5],
    "farneback_params": {
        "pyr_scale": [0.3, 0.4, 0.5],
        "levels": [1, 2, 3, 4],
        "winsize": [9, 15],
        "iterations": [3, 4, 5],
        "poly_n": [5, 7],
        "poly_sigma": [1.1, 1.2],
        "flags": [0],
        },
    }

    # Extract Farneback parameter combinations
    farneback_params = [
        dict(zip(search_space["farneback_params"].keys(), values))
        for values in product(*search_space["farneback_params"].values())
    ]

    # Combine with magnitude threshold
    configs = [
        {
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "max_flows" : MAX_FLOWS,
            "magnitude_thresh": mt,
            "farneback_params": fp,
        }
        for mt, fp in product(
            search_space["magnitude_thresh"], farneback_params
        )
    ]
 
    return configs
    

def create_dataloaders(config = CONFIG, batch_size = BATCH_SIZE):
    #initialize dataset
    dataset = Stacked3DDataset(DATA_DIR, ANNOTATIONS_DIR, config)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    indices = list(range(dataset_size))

    train_indices, temp_indices = train_test_split(indices, test_size=(val_size + test_size), random_state=30)
    val_indices, test_indices = train_test_split(temp_indices, test_size=test_size, random_state=30)

    #subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    #dataloaders
    train_loader = DataLoader(train_subset, batch_size, num_workers=8, shuffle = True, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size, num_workers=8, shuffle = False, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size, num_workers=4, shuffle = False, pin_memory=True)

    print("CREATED DATALOADERS!")
    print(f"Training set examples: {len(train_loader.dataset)}")
    print(f"Validation set examples: {len(val_loader.dataset)}")
    print(f"Test set examples: {len(test_loader.dataset)}")
    return train_loader, val_loader, test_loader

def hyperparameter_tuning(configs):
    
    #log results in file
    #select 100 random configs
    set_seed(29)
    configs = random.sample(configs, min(len(configs), 50))
    print("RANDOM CONFIGS SAMPLED: ", len(configs))
    best_acc = 0.0
    best_config = None
    for config in configs:
        print("TRYING CONFIG: ", config )
        train_loader, val_loader, _ = create_dataloaders(config, BATCH_SIZE)
        train_acc, val_acc = train_validate(train_loader, val_loader, epochs = 10)
        if val_acc > best_acc:
            best_acc = val_acc
            best_config = config
    
        print("VAL ACCURACY: ", val_acc)
        with open(LOG_VAL, "a") as log_file:
            log_file.write(f"\nConfiguration: {config} \nTrain accuracy: {train_acc} \nValidation_accuracy: {val_acc} \n")

    print("BEST CONFIG:d ", best_config)
    print("ACCURACY: ", best_acc)

    with open(LOG_VAL, "a") as log_file:
        log_file.write(f"\n\nBest configuration: {best_config}, Validation accuracy: {best_acc}")


def train_validate(train_loader, val_loader, epochs = EPOCHS):
    model = Gesture3DNet().to(device)
    optimizer = Adam(params = model.parameters(), lr = 3e-5, weight_decay = 1e-6)
    criterion = CrossEntropyLoss()
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        running_accuracy = 0.0

        train_loader_iter = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in train_loader_iter:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss
            _, predicted = torch.max(outputs,1)
            accuracy = torch.mean(predicted == labels, dtype=torch.float32)
            running_accuracy += accuracy
            
            #update progress bar
            train_loader_iter.set_postfix(loss=loss.item(), accuracy=(predicted == labels).float().mean().item())   

        avg_loss = running_loss / len(train_loader)
        epoch_accuracy = running_accuracy / len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        validation_misclassified_samples = []
        if (epoch + 1) % 5 == 0:
            #evaluate on validation set every 5 epochs
            running_accuracy = 0.0
            running_loss = 0.0

            with torch.no_grad():
                model.eval()
                for inputs, labels in tqdm.tqdm(val_loader, desc=f"[Validation]"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss
                    _, predicted = torch.max(outputs,1)
                    accuracy = torch.mean(predicted == labels, dtype=torch.float32)
                    running_accuracy += accuracy

                    misclassified = (predicted != labels)
                    if misclassified.any():
                        validation_misclassified_samples.extend(
                            list(zip(labels[misclassified].cpu().numpy(), predicted[misclassified].cpu().numpy()))
                        )

            avg_loss = running_loss / len(val_loader)
            val_accuracy = running_accuracy / len(val_loader)
            print(f"Validation loss: {avg_loss.item():.4f}")
            print(f"Validation accuracy: {val_accuracy.item():.4f}")
            print(f"Validation Misclassified Samples (label, prediction): {validation_misclassified_samples}")
            if val_accuracy > best_val_acc:
                print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_accuracy:.4f}")
                best_val_acc = val_accuracy

    return epoch_accuracy, best_val_acc

def set_seed(seed=30):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    print(f"Running on : {device}")
    set_seed(SEED)
    hyperparameter_tuning(generate_configs())
    # train_loader, val_loader, _ = create_dataloaders(batch_size = BATCH_SIZE)
    # train_validate(train_loader, val_loader)