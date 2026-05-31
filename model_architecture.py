#model_architecture

import sys
# pip install torchvision
# {sys.executable} -m pip uninstall -y torch torchvision torchaudio
# {sys.executable} -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# import torchvision


# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns


class InitialCNN(nn.Module):
    def __init__(self, improvement=False):
        super().__init__()
        self.improvement = improvement

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        if self.improvement:
            self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):

        # --- LAYER 1 ---
        if self.improvement:
            x = self.bn1(self.conv1(x))
            x = torch.relu(x)
        else:
            x = torch.sigmoid(self.conv1(x))

        x = F.max_pool2d(x, 2)

        # --- LAYER 2 ---
        if self.improvement:
             x = torch.relu(self.conv2(x))
        else:
             x = torch.sigmoid(self.conv2(x))

        x = F.max_pool2d(x, 2)

        # --- LAYER 3 ---
        if self.improvement:
             x = torch.relu(self.conv3(x))
        else:
             x = torch.sigmoid(self.conv3(x))

        x = F.max_pool2d(x, 2)

        # --- LAYER 4 ---
        if self.improvement:
             x = torch.relu(self.conv4(x))
        else:
             x = torch.sigmoid(self.conv4(x))

        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        # --- FC LAYER 1 ---
        if self.improvement:
             x = torch.relu(self.fc1(x))
        else:
             x = torch.sigmoid(self.fc1(x))

        x = self.fc2(x)

        return x


class InitialMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Model:

    def __init__(self, model, augmentation=False):


        # -------------------------
        # Device + Seed
        # -------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # -------------------------
        # Model
        # -------------------------
        self.model = model.to(self.device)

        # -------------------------
        # Data
        # -------------------------
        self.augmentation = augmentation

        if augmentation is False:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))
            ])

        elif augmentation is True:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    32,
                    scale=(0.7, 0.8),   # zoom range
                    ratio=(0.75, 1.33)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))
            ])

        train_dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )


        # FAST VERSION (small subset)
        train_dataset = torch.utils.data.Subset(train_dataset, range(25))
        test_dataset = torch.utils.data.Subset(test_dataset, range(5))

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        #Create DataLoaders
        batch_size = 128

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )


    # =========================================================
    # TRAIN
    # =========================================================
    def train(self):

        self.model.train()

        running_loss = 0
        correct = 0
        total = 0

        gradient_norms = {}

        for images, labels in self.train_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()

            # gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:

                    grad_norm = param.grad.norm().item()

                    if name not in gradient_norms:
                        gradient_norms[name] = []

                    gradient_norms[name].append(grad_norm)

            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100 * correct / total
        avg_loss = running_loss / len(self.train_loader)


        return avg_loss, acc, gradient_norms

    # =========================================================
    # TEST
    # =========================================================
    def test(self):

        self.model.eval()

        running_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in self.test_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = 100 * correct / total
        avg_loss = running_loss / len(self.test_loader)

        return avg_loss, acc

    # =========================================================
    # FULL TRAINING LOOP
    # =========================================================
    def evaluate(self, epochs=15, patience=False, weight_decay=0):

        train_losses, test_losses = [], []
        train_accs, test_accs = [], []

        grad_history = {}
        epochs_durations = []

        # --- Early Stopping Trackers ---
        best_test_loss = float('inf')
        epochs_without_improvement = 0

        # -------------------------
        # Loss + Optimizer
        # -------------------------
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=weight_decay)

        for epoch in range(epochs):

            start_time = time.time()
            train_loss, train_acc, grads = self.train()
            end_time = time.time()

            training_time = end_time - start_time
            epochs_durations.append(training_time)

            test_loss, test_acc = self.test()

            grad_history = grads

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            print(f"Epoch {epoch+1}/{epochs} | Time {training_time:.2f} sec")
            print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
            print("-" * 40)

            # --- TRUE EARLY STOPPING LOGIC ---
            if patience is not False:
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    epochs_without_improvement = 0 # Reset counter if we improve
                else:
                    epochs_without_improvement += 1 # Increment if we got worse

                # If we haven't improved in 'patience' number of epochs, stop
                if epochs_without_improvement >= patience:
                    print(f"🛑 EARLY STOPPING TRIGGERED at Epoch {epoch+1}!")
                    print(f"Test loss has not improved for {patience} epochs.")
                    break

        return train_losses, test_losses, train_accs, test_accs, grad_history, epochs_durations

    # =========================================================
    # VISUALIZATION
    # =========================================================

    def plot_learning_curves(self, train_losses, test_losses, train_accs, test_accs):

        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(12,5))

        # -------------------------
        # Loss plot
        # -------------------------
        plt.subplot(1,2,1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, test_losses, label="Test Loss")
        plt.xlabel("Epoch")              # x-axis
        plt.ylabel("Loss")               # y-axis
        plt.legend()
        plt.title("Loss over Epochs")

        # -------------------------
        # Accuracy plot
        # -------------------------
        plt.subplot(1,2,2)
        plt.plot(epochs, train_accs, label="Train Accuracy")
        plt.plot(epochs, test_accs, label="Test Accuracy")
        plt.xlabel("Epoch")              # x-axis
        plt.ylabel("Accuracy (%)")       # y-axis
        plt.legend()
        plt.title("Accuracy over Epochs")

        plt.show()



    def plot_gradients(self, grad_history):

        plt.figure(figsize=(10,6))

        for name, vals in grad_history.items():
            plt.plot(vals, label=name)

        plt.xlabel("Batch / Iteration Step")   # x-axis
        plt.ylabel("Gradient Norm")           # y-axis
        plt.title("Gradient Magnitude per Layer")
        plt.legend()

        plt.show()


    def show_samples(self):

        base_dataset = self.train_dataset.dataset if hasattr(self.train_dataset, "dataset") else self.train_dataset
        classes = base_dataset.classes

        images, labels = next(iter(self.train_loader))

        fig, axes = plt.subplots(2, 5, figsize=(10,5))

        for i, ax in enumerate(axes.flat):

            img = images[i] / 2 + 0.5
            img = img.permute(1,2,0)

            ax.imshow(img)
            ax.set_title(classes[labels[i]])
            ax.axis("off")

        plt.show()


    def print_model_report(self, epochs_durations):
        """
        Prints:
        - number of parameters
        - hyperparameters
        - training speed
        - memory usage estimate
        """

        # -------------------------
        # 1. Number of parameters
        # -------------------------
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # -------------------------
        # 2. Hyperparameters
        # -------------------------
        optimizer_name = type(self.optimizer).__name__
        lr = self.optimizer.param_groups[0]['lr']
        batch_size = self.train_loader.batch_size

        # -------------------------
        # 3. Training speed
        # -------------------------
        avg_epoch_time = round(np.mean(epochs_durations),2)

        # -------------------------
        # 4. Memory estimation (rough)
        # -------------------------
        param_memory = total_params * 4 / (1024 ** 2)  # float32 = 4 bytes
        grad_memory = total_params * 4 / (1024 ** 2)
        optimizer_memory = total_params * 8 / (1024 ** 2)  # Adam approx (m + v)

        total_memory_mb = param_memory + grad_memory + optimizer_memory

        # -------------------------
        # PRINT REPORT
        # -------------------------
        print("\n" + "="*50)
        print("MODEL REPORT")
        print("="*50)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        print("\nHyperparameters:")
        print(f"Optimizer: {optimizer_name}")
        print(f"Learning rate: {lr}")
        print(f"Batch size: {batch_size}")

        print("\nTraining speed:")
        print(f'{avg_epoch_time} seconds per epoch on average')

        print("\nEstimated memory usage:")
        print(f"Parameters memory: {param_memory:.2f} MB")
        print(f"Gradients memory: {grad_memory:.2f} MB")
        print(f"Optimizer states memory: {optimizer_memory:.2f} MB")
        print(f"Total estimated: {total_memory_mb:.2f} MB")

        print("="*50 + "\n")


    def plot_confusion_matrix(self):

        self.model.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in self.test_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = outputs.max(1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix (CIFAR-10)")
        plt.show()