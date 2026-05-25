import time
import os
import sys
import copy
import numpy as np
import torch
import torch.optim as optim
import awkward as ak

def normalize(dataset):
    # Better using the mean() and std() of the training dataset on all datasets. In this case, the model trains on different scales.
    """
    Normalize the input labels (time, x, y) for a given dataset. 
    This is done, by splitting the dataset in its labels, normalizing and concatenating them back together in the end.

    Parameters:
        dataset (3D-ak-array): The input dataset to be normalized

    Returns:
        ak-array: The three split normalized labels
    """
    times = dataset["data"][:, 0:1, :]  # important to index the time dimension with 0:1 to keep this dimension (n_events, 1, n_hits)
                                        # with [:,0,:] we would get a 2D array of shape (n_events, n_hits)
    x = dataset["data"][:, 1:2, :]
    y = dataset["data"][:, 2:3, :]

    norm_times = (times - ak.mean(times))/ak.std(times)
    norm_x = (x - ak.mean(x))/ak.std(x)
    norm_y = (y - ak.mean(y))/ak.std(y)

    # Concatenate the normalized data back together
    dataset["data"] = ak.concatenate([norm_times, norm_x, norm_y], axis=1)

    # Normalize labels (this can be done in-place), e.g. by
    # Calculating the normalization stats for xpos and ypos, so that the mean() and std() functions are not called multiple times
    xpos_mean = ak.mean(dataset["xpos"])
    xpos_std = ak.std(dataset["xpos"])
    ypos_mean = ak.mean(dataset["ypos"])
    ypos_std = ak.std(dataset["ypos"])

    dataset["xpos"] = (dataset["xpos"] - xpos_mean) / xpos_std
    dataset["ypos"] = (dataset["ypos"] - ypos_mean) / ypos_std

    # Saving the normalization data (not for time, x and y, since we only need to denormalize the target labels (xpos and ypos)
    norm_stats = {
        'x_mean': xpos_mean,
        'x_std': xpos_std,
        'y_mean': ypos_mean,
        'y_std': ypos_std
    }

    return dataset, norm_stats

def denormalize(predictions, norm_stats):
    """
    Denormalize the target labels (xpos, ypos) after the training. 

    Parameters:
        predictions (np.array): The target labels to be normalized
        norm_stats (dict): The used mean and std values used for the initial normalization

    Returns:
        np.array: The three split normalized labels
    """
    denorm_xpos = predictions[:, 0] * norm_stats['x_std'] + norm_stats['x_mean']
    denorm_ypos = predictions[:, 1] * norm_stats['y_std'] + norm_stats['y_mean']
    denorm_predictions = np.stack([denorm_xpos, denorm_ypos], axis=1)

    return denorm_predictions

def train_model(
    model,
    train_loader,
    val_loader,
    loss_function,
    learning_rate,
    num_epochs,
    patience,
    device,
    plot_fn=None,
    plot_interval=10,
    model_name=None,
):
    """
    Trains a given model using the provided training and validation data loaders, loss function, and optimizer.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to be trained.
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        val_loader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        loss_function : torch.nn.Module
            Loss function to be used for training.
        learning_rate : float
            learning rate
        num_epochs : int
            Number of epochs to train the model.
        patience : int
            Number of epochs with no improvement after which training will be stopped.
        device : torch.device
            Device on which to perform training (e.g., 'cpu' or 'cuda').
        plot_fn : callable, optional
            Function to plot the model predictions during training. Default is None.
        plot_interval : int, optional
            Interval at which to plot the model predictions during training. Default is 10.
        plot_kwargs : dict, optional
            Additional keyword arguments to be passed to the plot function. Default is None.
        model_name : str, optional
            Name of the model for saving the best model. Default is None.
            If provided, the best model will be saved to the "models" directory with the given name.

        Returns
        -------
        tuple
            A tuple containing two lists:
            - train_losses (list of float): List of average training losses for each epoch.
            - val_losses (list of float): List of average validation losses for each epoch.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model = None

    for epoch in range(num_epochs):
        start_time = time.time()  # Start the timer for this epoch

        # Training phase
        model.train()
        total_train_loss = 0.0
        for step, (batch_graphs, batch_labels) in enumerate(train_loader):
            optimizer.zero_grad()

            predictions = model(batch_graphs)
            loss = loss_function(predictions, batch_labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Print progress every 10th step, updating the same line
            if (step + 1) % 10 == 0:
                sys.stdout.write(f"\rEpoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                sys.stdout.flush()

        sys.stdout.write("\n")  # Move to the next line after the epoch

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_graphs, batch_labels in val_loader:

                predictions = model(batch_graphs)
                val_loss = loss_function(predictions, batch_labels)

                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        # Store losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Print epoch summary
        epoch_time = time.time() - start_time  # Calculate epoch time
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f} seconds")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model = copy.copy(model.state_dict())
            # Save the best model to the "models" directory
            if not os.path.exists("models"):
                os.makedirs("models")
            if model_name is not None:
                torch.save(best_model, f"models/{model_name}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        if epoch % plot_interval == 0:
            if plot_fn is not None:
                plot_fn(
                    model,
                    loss_function,
                    device,
                    train_losses,
                    val_losses,
                    suffix="epoch_%.5d" % epoch,
                )

    return train_losses, val_losses, best_model


def evaluate_model(model, test_loader, loss_function, device):
    """
    Evaluate the given model on the test dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to evaluate.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    loss_function : callable
        Loss function used to compute the loss.
    device : torch.device
        Device on which to perform computations (e.g., 'cpu' or 'cuda').

    Returns
    -------
    all_predictions : numpy.ndarray
        Array of denormalized predictions made by the model.
    all_true_labels : numpy.ndarray
        Array of denormalized true labels from the test dataset.
    """
    print("Evaluating model on the test dataset...")
    model.eval()
    total_test_loss = 0.0
    all_predictions = []
    all_true_labels = []

    first_batch_graphs = None
    first_batch_labels = None

    with torch.no_grad():
        for batch_index, (batch_graphs, batch_labels) in enumerate(test_loader):
            predictions = model(batch_graphs)

            test_loss = loss_function(predictions, batch_labels)

            total_test_loss += test_loss.item()
            all_predictions.append(predictions.cpu())
            all_true_labels.append(batch_labels.cpu())

            if batch_index == 0:
                first_batch_graphs = batch_graphs
                first_batch_labels = batch_labels

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss:.4f}")
    return torch.cat(all_predictions).numpy(), torch.cat(all_true_labels).numpy(), first_batch_graphs, first_batch_labels
