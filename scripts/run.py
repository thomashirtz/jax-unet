import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.training import train_state
from jax import random


def create_train_state(rng, model, learning_rate, input_shape):
    """
    Initialize the training state including model parameters and optimizer.

    Args:
        rng (jax.random.PRNGKey): Random number generator key for initializing the model.
        model (nn.Module): The model to be trained.
        learning_rate (float): Learning rate for the optimizer.
        input_shape (tuple): Shape of the input tensor (batch_size, channels, height, width).

    Returns:
        train_state.TrainState: A train state containing the initialized parameters, optimizer, and apply function.
    """
    params = model.init(rng, jnp.ones(input_shape))  # Initialize model parameters
    tx = optax.adam(learning_rate)  # Adam optimizer
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def compute_loss(params, apply_fn, images, labels):
    """
    Compute mean squared error (MSE) between predictions and labels.

    Args:
        params (dict): Model parameters.
        apply_fn (function): Apply function for the model.
        images (jnp.ndarray): Batch of input images.
        labels (jnp.ndarray): Ground truth labels.

    Returns:
        jnp.ndarray: Scalar loss value (MSE).
    """
    preds = apply_fn(params, images)
    return jnp.mean((preds - labels) ** 2)


@jax.jit
def train_step(state, images, labels):
    """
    Perform a single training step: forward pass, gradient calculation, and parameter update.

    Args:
        state (train_state.TrainState): Current training state.
        images (jnp.ndarray): Batch of input images.
        labels (jnp.ndarray): Batch of ground truth labels.

    Returns:
        Tuple[train_state.TrainState, jnp.ndarray]: Updated training state and loss value.
    """
    def loss_fn(params):
        return compute_loss(params, state.apply_fn, images, labels)

    grads = jax.grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    loss = loss_fn(state.params)
    return new_state, loss


def train_model(model, rng, input_shape, num_epochs=100, batch_size=16, learning_rate=1e-4):
    """
    Train the U-Net model on random generated data.

    Args:
        model (nn.Module): The U-Net model to be trained.
        rng (jax.random.PRNGKey): Random key for initializing the model and dataset.
        input_shape (tuple): Shape of input images (batch_size, channels, height, width).
        num_epochs (int, optional): Number of training epochs. Defaults to 100.
        batch_size (int, optional): Batch size for training. Defaults to 16.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 1e-4.

    Returns:
        train_state.TrainState: Final model state after training.
    """
    rng, init_rng = random.split(rng)
    state = create_train_state(init_rng, model, learning_rate, input_shape)

    # Generate random dataset
    num_samples = 100
    _, channels, height, width = input_shape
    images = jax.random.uniform(rng, (num_samples, channels, height, width))
    labels = jax.random.uniform(rng, (num_samples, model.out_channels, height, width))

    # Training loop
    for epoch in range(num_epochs):
        batch_indices = np.random.choice(num_samples, batch_size)
        batch_images = images[batch_indices]
        batch_labels = labels[batch_indices]

        state, loss = train_step(state, batch_images, batch_labels)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return state


def visualize_results(state, model, rng, input_shape):
    """
    Visualize the input image and corresponding model predictions.

    Args:
        state (train_state.TrainState): Model state containing trained parameters.
        model (nn.Module): Trained U-Net model.
        rng (jax.random.PRNGKey): Random key for generating test data.
        input_shape (tuple): Shape of the input image (channels, height, width).
    """
    test_image = jax.random.uniform(rng, input_shape)
    test_image = jnp.expand_dims(test_image, 0)  # Add batch dimension

    preds = model.apply(state.params, test_image)

    # Convert tensors to numpy for visualization
    test_image_np = np.array(test_image[0])
    preds_np = np.array(preds[0])

    # Plot input image and prediction
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(test_image_np[0, :, :], cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Output")
    plt.imshow(preds_np[0, :, :], cmap="gray")
    plt.show()


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)

    # Define model parameters
    in_channels = 1  # Grayscale images
    out_channels = 1  # Output segmentation mask
    input_shape = (16, in_channels, 64, 64)  # Batch of 16, 64x64 images

    # Initialize U-Net model
    model = UNetwork(
        in_channels=in_channels,
        out_channels=out_channels,
        channel_list=[16, 32, 64],
        bilinear=False
    )

    # Train the model
    state = train_model(model, rng, input_shape, num_epochs=100, batch_size=16, learning_rate=1e-3)

    # Visualize the results
    visualize_results(state, model, rng, input_shape[1:])
