from model_architecture import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ==========================================
        # PHASE 1: CNN Feature Extractor
        # ==========================================
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # After 3 pooling layers (32/2/2/2), spatial size is 4x4.
        # We have 128 channels, so the flattened size is 128 * 4 * 4 = 2048

        # ==========================================
        # PHASE 2: MLP Classifier
        # ==========================================
        self.fc1 = nn.Linear(2048, 512)
        self.dropout1 = nn.Dropout(0.5) # Drops 50% of neurons to prevent overfitting

        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):

        # --- Feature Extraction (CNN) ---
        x = self.bn1(self.conv1(x))
        x = torch.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.bn2(self.conv2(x))
        x = torch.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.bn3(self.conv3(x))
        x = torch.relu(x)
        x = F.max_pool2d(x, 2)

        # --- Bridge: Flattening ---
        x = x.view(x.size(0), -1)

        # --- Global Classification (MLP) ---
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x

print('\n' + '='*50)
print('TASK 5: HYBRID CNN-MLP MODEL (Augmented + Early Stop)')
print('='*50)

# Initialize the new Hybrid Model WITH Augmentation
hybrid_trainer = Model(HybridModel(), augmentation=True)

# Evaluate using patience-based Early Stopping (stops if test loss stagnates for 4 epochs)
hybrid_results = hybrid_trainer.evaluate(epochs=25, patience=4)

# Print the report and visualize
hybrid_trainer.print_model_report(hybrid_results[-1])
hybrid_trainer.plot_learning_curves(*hybrid_results[:4])
hybrid_trainer.plot_confusion_matrix()





print('\n' + '='*50)
print('CREATIVITY TASK: HYPERPARAMETER TUNING (Learning Rate)')
print('='*50)

# Test 1: A Higher Learning Rate (0.01 instead of 0.001)
print('Testing High Learning Rate (lr=0.01)...')
high_lr_trainer = Model(HybridModel(), augmentation=True)
results_high = high_lr_trainer.evaluate(epochs=15, patience=4, lr=0.01)
high_lr_trainer.plot_learning_curves(*results_high[:4])

# Test 2: A Lower Learning Rate (0.0001 instead of 0.001)
print('Testing Low Learning Rate (lr=0.0001)...')
low_lr_trainer = Model(HybridModel(), augmentation=True)
results_low = low_lr_trainer.evaluate(epochs=15, patience=4, lr=0.0001)
low_lr_trainer.plot_learning_curves(*results_low[:4])




print('\n' + '='*50)
print('CREATIVITY TASK: HYBRID MLP WEIGHT VISUALIZATION')
print('='*50)

# Extract the weights from the first Linear layer (fc1) of the trained Hybrid Model
# Shape of fc1.weight is [512, 2048]
hybrid_fc1_weights = hybrid_trainer.model.fc1.weight.data.cpu().numpy()

# We will just plot the weights for the first 10 neurons in this layer
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Hybrid Model: Weights of first 10 Neurons in FC1\n(Mapping 2048 abstract features, NOT pixels)", fontsize=14)

for i, ax in enumerate(axes.flat):
    # Get the 2048 weights for neuron 'i'
    neuron_weights = hybrid_fc1_weights[i]

    # Reshape it into a 32x64 grid simply so we can view it as a 2D image
    # (Note: This shape is purely for visualization, it has no spatial meaning)
    weight_img = neuron_weights.reshape(32, 64)

    im = ax.imshow(weight_img, cmap='viridis')
    ax.set_title(f"Neuron {i+1}")
    ax.axis('off')

plt.tight_layout()
plt.show()