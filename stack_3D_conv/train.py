from two_stream_model import SpatialStream, TemporalStream
from stack_3d_dataset import Stacked3DDataset
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.nn import CrossEntropyLoss
from torch.cuda import is_available
import numpy as np
import random
import tqdm
import os
from collections import Counter

device = "cuda" if is_available() else "cpu"
torch.cuda.empty_cache()

SEED = 42
SPATIAL_STREAM_PATH = "models/spatial_stream.pt"
TEMPORAL_STREAM_PATH = "models/temporal_stream.pt"
DATA_DIR = "rec_dataset2"
ANNOTATIONS_DIR = "rec_dataset2/data.csv"
NUM_CLASSES = 7
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
MAX_FLOWS = 30
CONFIG = {
    "image_width" : IMAGE_WIDTH,
    "image_height" : IMAGE_HEIGHT,
    "magnitude_thresh" : 1,
    "max_flows" : MAX_FLOWS,
    "farneback_params" : {
        "pyr_scale": 0.4,
        "levels": 4,
        "winsize": 15,
        "iterations": 5,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }
}

EPOCHS = 100
BATCH_SIZE = 4

def count_class_distribution(dataset):
    class_counts = Counter()
    
    for _, label in dataset:
        class_counts[label.item()] += 1
    
    return class_counts

def create_dataloaders(config = CONFIG, batch_size = BATCH_SIZE, spatial_stream = False, test_split = .1, count = False):
    #initialize dataset
    dataset = Stacked3DDataset(DATA_DIR, ANNOTATIONS_DIR, spatial_stream, config, augment=True)
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)

    indices = list(range(dataset_size))
    train_indices, test_indices = train_test_split(indices, test_size = test_size, random_state=30)

    #subsets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    if count:
        #count class distributions
        def count_classes(subset):
            class_counts = Counter()
            for _, label in subset:
                class_counts[label] += 1
            return class_counts

        train_class_distribution = count_classes(train_subset)
        test_class_distribution = count_classes(test_subset)

        print("Class Distribution in Training Set:", train_class_distribution)
        print("Class Distribution in Testing Set:", test_class_distribution)

    #dataloaders
    train_loader = DataLoader(train_subset, batch_size, num_workers=4, shuffle = True, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size, num_workers=4, shuffle = False, pin_memory=True)

    print("CREATED DATALOADERS!")
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")
    return train_loader, test_loader 

def train(train_loader, test_loader, spatial, base_lr, max_lr, weight_decay, epochs=EPOCHS):
    if spatial:
        model_path = SPATIAL_STREAM_PATH
        model = SpatialStream(NUM_CLASSES).to(device)
    else:
        model_path = TEMPORAL_STREAM_PATH
        model = TemporalStream(NUM_CLASSES).to(device)

    if os.path.exists(model_path):
        print("Found model, loading state dict")
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_test_acc = checkpoint['accuracy']
        print(f"Loaded model with best accuracy: {best_test_acc:.4f}")
    else:
        print("Model not found. Training from scratch.")
        best_test_acc = 0.0

    optimizer = Adam(params=model.parameters(), lr=base_lr, weight_decay=weight_decay)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scheduler = CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=len(train_loader) // 2,
        step_size_down=len(train_loader) // 2,
        mode='triangular',
        cycle_momentum=False #False for Adam
    )
    
    criterion = CrossEntropyLoss()
    accumulation_steps = 8  #simulates a larger batch size

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        running_accuracy = 0.0
        num_batches = 0

        current_lr = optimizer.param_groups[0]['lr']
        optimizer.zero_grad()

        train_loader_iter = tqdm.tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{epochs}")
        for i, (inputs, labels) in enumerate(train_loader_iter):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels) 
            loss = loss / accumulation_steps  #normalize for accumulation
            loss.backward()

            #perform optimizer step after accumulation_steps
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step() #for cyclic LR  

            running_loss += loss.item() * accumulation_steps  #denormalize
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).float().mean().item()
            running_accuracy += accuracy
            num_batches += 1

            #update progress bar
            train_loader_iter.set_postfix(loss=loss.item() * accumulation_steps, accuracy=accuracy, lr = current_lr)

        avg_loss = running_loss / num_batches
        epoch_accuracy = running_accuracy / num_batches
        print(f"[Train] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        if (epoch + 1) % 3 == 0 or (epoch + 1) == EPOCHS:
            test_running_loss = 0.0
            test_running_accuracy = 0.0
            # test_misclassified_samples = []
            num_test_batches = 0

            model.eval()
            with torch.no_grad():
                for inputs, labels in tqdm.tqdm(test_loader, desc=f"[Test]"):
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    test_running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    accuracy = (predicted == labels).float().mean().item()
                    test_running_accuracy += accuracy
                    num_test_batches += 1

                    # misclassified = (predicted != labels)
                    # if misclassified.any():
                    #     test_misclassified_samples.extend(
                    #         list(zip(labels[misclassified].cpu().numpy(), predicted[misclassified].cpu().numpy()))
                    #     )

                avg_test_loss = test_running_loss / num_test_batches
                avg_test_accuracy = test_running_accuracy / num_test_batches
                # scheduler.step(avg_test_loss)

            print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")
            # print(f"Misclassified Samples (label, prediction): {test_misclassified_samples}")

            # Save best model
            if avg_test_accuracy > best_test_acc:
                print(f"Test accuracy improved from {best_test_acc:.4f} to {avg_test_accuracy:.4f}. Saving model...")
                best_test_acc = avg_test_accuracy
                torch.save({'model_state_dict': model.state_dict(), 'accuracy': best_test_acc}, model_path)

    return epoch_accuracy, best_test_acc

def evaluate_model(test_loader, spatial):

    if spatial:
        model_path = SPATIAL_STREAM_PATH
        model = SpatialStream(NUM_CLASSES).to(device)
    else:
        model_path = TEMPORAL_STREAM_PATH
        model = TemporalStream(NUM_CLASSES).to(device)

    if os.path.exists(model_path):
        print("Found model, loading state dict")
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_test_acc = checkpoint['accuracy']
        print(f"Loaded model with best accuracy: {best_test_acc:.4f}")
    else:
        print("Model not found. Exiting")
        return

    running_accuracy = 0.0
    running_loss = 0.0
    test_misclassified_samples =[]

    criterion = CrossEntropyLoss()
    with torch.no_grad():
        model.eval()
        for inputs, labels in tqdm.tqdm(test_loader, desc=f"[Test]"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss
            _, predicted = torch.max(outputs,1)
            accuracy = torch.mean(predicted == labels, dtype=torch.float32)
            running_accuracy += accuracy

            misclassified = (predicted != labels)
            if misclassified.any():
                test_misclassified_samples.extend(
                    list(zip(labels[misclassified].cpu().numpy(), predicted[misclassified].cpu().numpy()))
                )

    avg_loss = running_loss / len(test_loader)
    val_accuracy = running_accuracy / len(test_loader)
    print(f"Test loss: {avg_loss.item():.4f}")
    print(f"Test accuracy: {val_accuracy.item():.4f}")
    print(f"Test Misclassified Samples (label, prediction): {test_misclassified_samples}")



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    print(f"Running on : {device}")
    set_seed(SEED)

    # train spatial stream
    train_loader, test_loader = create_dataloaders(CONFIG, BATCH_SIZE, spatial_stream=True, test_split=0.2)
    # train(train_loader, test_loader, spatial=True, base_lr = 1e-5, max_lr = 1e-3,  weight_decay=1e-5)
    # evaluate_model(test_loader, spatial=True)

    #train temporal stream
    train_loader, test_loader = create_dataloaders(CONFIG, batch_size=2, spatial_stream=False, test_split=0.2)
    # train(train_loader, test_loader, spatial=False, base_lr = 6e-6, max_lr = 1e-3, weight_decay=0)
    # evaluate_model(test_loader, spatial=False)
    