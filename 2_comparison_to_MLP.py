from model_architecture import Model, InitialCNN, InitialMLP
import matplotlib.pyplot as plt

#MLP Comparison
# =========================================================
# VISUALIZE CNN FILTERS + MLP WEIGHTS
# =========================================================

def visualize_cnn_filters(model, max_filters=16):

    """
    Visualize learned filters from the first CNN layer
    """

    # get conv1 weights
    filters = model.conv1.weight.data.cpu()

    num_filters = min(filters.shape[0], max_filters)

    fig, axes = plt.subplots(4, 4, figsize=(8,8))

    for i, ax in enumerate(axes.flat):

        if i >= num_filters:
            break

        # shape = [3, 3, 3]
        filt = filters[i]

        # normalize for display
        filt = (filt - filt.min()) / (filt.max() - filt.min())

        # convert CHW -> HWC
        filt = filt.permute(1, 2, 0)

        ax.imshow(filt)
        ax.set_title(f"Filter {i+1}")
        ax.axis("off")

    plt.suptitle("CNN Learned Filters (conv1)")
    plt.tight_layout()
    plt.show()



def visualize_mlp_weights(model, max_neurons=16):

    """
    Visualize learned weights from first MLP layer
    """

    # get fc1 weights
    weights = model.fc1.weight.data.cpu()

    num_neurons = min(weights.shape[0], max_neurons)

    fig, axes = plt.subplots(4, 4, figsize=(8,8))

    for i, ax in enumerate(axes.flat):

        if i >= num_neurons:
            break

        # reshape into image
        img = weights[i].reshape(32, 32, 3)

        # normalize for display
        img = (img - img.min()) / (img.max() - img.min())

        ax.imshow(img)
        ax.set_title(f"Neuron {i+1}")
        ax.axis("off")

    plt.suptitle("MLP Learned Weights (fc1)")
    plt.tight_layout()
    plt.show()



# =========================================================
# 1. TRAIN AND EVALUATE CNN
# =========================================================
print('--- Training Improved CNN ---')
cnn_trainer = Model(InitialCNN(improvement=True))
cnn_results = cnn_trainer.evaluate()

# Print the report to get parameters, speed, and memory for your PDF
cnn_trainer.print_model_report(cnn_results[-1])

# =========================================================
# 2. TRAIN AND EVALUATE MLP
# =========================================================
print('\n--- Training MLP ---')
mlp_trainer = Model(InitialMLP())
mlp_results = mlp_trainer.evaluate()

# Print the report to compare against the CNN
mlp_trainer.print_model_report(mlp_results[-1])
mlp_trainer.plot_learning_curves(*mlp_results[:4])

# =========================================================
# 3. VISUALIZE AND COMPARE WEIGHTS
# =========================================================
print('\n--- Visualization of Weights ---')

print('MLP First Layer Weights (fc1):')
visualize_mlp_weights(mlp_trainer.model)

print('CNN First Layer Filters (conv1):')
visualize_cnn_filters(cnn_trainer.model) # Now properly passing the CNN model!
