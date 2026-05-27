from model_architecture import Model, InitialCNN, InitialMLP
import matplotlib.pyplot as plt

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



print('Model CNN:')
trainer = Model(InitialCNN())
results = trainer.evaluate()
trainer.plot_learning_curves(*results[:4])
trainer.plot_gradients(results[4])
trainer.show_samples()
visualize_cnn_filters(trainer.model)

print('////////////////////////////////////////////////////////////////////////////////////////////////////')

print('Model MLP:')
trainer = Model(InitialMLP())
results = trainer.evaluate()
trainer.plot_learning_curves(*results[:4])
trainer.plot_gradients(results[4])
trainer.show_samples()
visualize_mlp_weights(trainer.model)


