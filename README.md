# Neural Networks - Second Task

This project trains CIFAR-10 classifiers with PyTorch. The shared code is in `model_architecture.py`; the numbered scripts run the assignment parts in order. `Final Report.pdf` contains the written explanation, results, figures, and conclusions.

## How to run

Run scripts from this folder so CIFAR-10 is downloaded/read from `./data`:

```bash
python 1_CNN_debugging.py
python 2_comparison_to_MLP.py
python 3_overfitting_analysis.py
python 4_augmentation.py
python 5_creativity_task.py
```

Each script trains models and opens plots with Matplotlib. Training may take time, especially without a GPU.

## Code guide

- `model_architecture.py` contains the reusable building blocks.
  - `InitialCNN(improvement=False)` is the original CNN. With `improvement=True`, it uses BatchNorm on the first convolution and ReLU activations instead of sigmoid.
  - `InitialMLP()` is a baseline fully connected network for flattened CIFAR-10 images.
  - `Model(model, augmentation=False)` wraps a PyTorch model with CIFAR-10 loading, training, testing, plotting, reports, and confusion matrices.
  - `Model.evaluate(epochs=15, patience=False, lr=0.001, weight_decay=0)` is the main training loop. It returns train/test losses, accuracies, gradient history, and epoch durations.
  - Useful visualization/report methods: `plot_learning_curves`, `plot_gradients`, `show_samples`, `print_model_report`, and `plot_confusion_matrix`.

- `1_CNN_debugging.py` runs the original CNN, then the improved CNN, and compares learning curves, gradients, and image samples.

- `2_comparison_to_MLP.py` trains the improved CNN and the MLP, prints model reports, plots MLP learning curves, and visualizes first-layer CNN filters and MLP weights.

- `3_overfitting_analysis.py` defines `LargerMLP`, trains it to show overfitting, then compares early stopping and weight decay as overfitting controls.

- `4_augmentation.py` compares the improved CNN with and without data augmentation, then plots learning curves, sample images, and confusion matrices.

- `5_creativity_task.py` defines `HybridModel`, a CNN feature extractor followed by an MLP classifier. It trains the hybrid model with augmentation and early stopping, compares learning-rate settings, and visualizes the hybrid classifier weights.

## Suggested reading order

Read `model_architecture.py`, then run the numbered scripts from `1` to `5`. Use `Final Report.pdf` alongside the scripts to connect each code result to the submitted explanation and figures.
