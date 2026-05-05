import time
import os
import sys
import copy
import numpy as np
import torch
import torch.optim as optim

def normalize(labels, p):
    """
    Normalize the input labels using percentile-based scaling.

    This function scales the input labels to a range of [0, 1] based on the specified percentiles.
    The scaling is done by computing the percentiles of the labels and then normalizing the labels
    using these percentile values.

    Parameters:
    labels (np.ndarray): The input labels to be normalized.
    p (float): The percentile value used for scaling. The function uses the p-th and (1-p)-th percentiles
               for normalization.

    Returns:
    tuple: A tuple containing the normalized labels and the range used for normalization.
           - normalized_labels (np.ndarray): The normalized labels.
           - ranges (np.ndarray): The range used for normalization, which includes the p-th and (1-p)-th percentiles.
    """
    ranges = np.percentile(labels, [100 * p, 100 * (1 - p)], axis=0)
    labels = (labels - ranges[0]) / (ranges[1] - ranges[0])
    return labels, ranges

# Function to denormalize the labels back to their original scale
def denormalize(labels, ranges):
    """
    Denormalize the input labels using the specified range.

    This function denormalizes the input labels using the specified range values.
    The denormalization is done by scaling the labels back to the original range
    using the provided range values.

    Parameters:
        labels (np.ndarray): The normalized labels to be  denormalized.
        ranges (np.ndarray): The range values used for normalization.

    Returns:
        np.ndarray: The denormalized labels.
    """
    return labels * (ranges[1] - ranges[0]) + ranges[0]

def denormalize_std(uncertainty, ranges):
    """
    Denormalizes the given uncertainty predictions using the provided range.

    It is different to the denormalization of the labels which also includes a shift.

    Parameters
    ----------
    uncertainty : array-like
        The normalized uncertainty to be denormalized.
    ranges : array-like
        A two-element array-like object where the first element is the minimum value
        and the second element is the maximum value of the original range.
    Returns
    -------
    array-like
        The denormalized uncertainty.
    """

    return uncertainty * (ranges[1] - ranges[0])

def get_normalized_data(data_path, return_SNR=False):
    """
    Load and normalize spectra and label data from the given path.
    Parameters
    ----------
    data_path : str
        The path to the directory containing the spectra and labels data files.
    Returns
    -------
    spectra : numpy.ndarray
        The normalized spectra data.
    labels : numpy.ndarray
        The normalized labels data (t_eff, log_g, fe_h).
    spectra_length : int
        The length of the spectra.
    n_labels : int
        The number of labels used (should be 3).
    labelNames : list of str
        The names of the labels used (t_eff, log_g, fe_h).
    ranges : numpy.ndarray
        The ranges used for normalization of the labels.
    """

    # Load the spectra data
    spectra = np.load(f"{data_path}/spectra.npy")
    spectra_length = spectra.shape[1]

    # Load the labels data
    # labels: mass, age, l_bol, dist, t_eff, log_g, fe_h
    labelNames = ["mass", "age", "l_bol", "dist", "t_eff", "log_g", "fe_h"]
    labels = np.load(f"{data_path}/labels.npy")
    SNR = labels[:, -1]
    labels = labels[:, :-1]

    # We only use the labels: t_eff, log_g, fe_h
    labelNames = labelNames[-3:]
    labels = labels[:, -3:]
    n_labels = labels.shape[1]

    labels, ranges = normalize(labels, 0.05)

    # Normalize spectra
    spectra = np.log(np.maximum(spectra, 0.2))

    if return_SNR:
        return spectra, labels, spectra_length, n_labels, labelNames, ranges, SNR
    return spectra, labels, spectra_length, n_labels, labelNames, ranges

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

    first_batch_spectra=None
    first_batch_labels=None

    with torch.no_grad():
        for batch_index, (batch_spectra, batch_labels) in enumerate(test_loader):
            batch_spectra, batch_labels = batch_spectra.to(device).unsqueeze(1), batch_labels.to(device)
            predictions = model(batch_spectra)

            test_loss = loss_function(batch_spectra, batch_labels,model)

            total_test_loss += test_loss.item()
            all_predictions.append(predictions.cpu())
            all_true_labels.append(batch_labels.cpu())

            if(batch_index==0):
                first_batch_spectra=batch_spectra
                first_batch_labels=batch_labels



    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss:.4f}")
    return torch.cat(all_predictions).numpy(), torch.cat(all_true_labels).numpy(), first_batch_spectra, first_batch_labels



