import numpy as np


def split_data(data, u, train_size, test_size, warm_up):
    """
    Split the system data and input signal into training and testing sets.

    Parameters
    ----------
    data : np.ndarray
        System output data with shape (output_size, total_time_steps).
        Typically generated from a dynamical system such as the Van der Pol oscillator.
        Example shape: (2, total_steps)

    u : np.ndarray
        Input control signal applied to the system.
        Example shape: (total_steps,)

    train_size : int
        Number of timesteps to be used for training.

    test_size : int
        Number of timesteps to be used for testing (free-running phase).

    warm_up : int
        Number of initial timesteps to discard during training (state stabilization).

    Returns
    -------
    train_data : np.ndarray
        Training output data for supervised learning.
        Shape: (output_size, train_size)

    test_data : np.ndarray
        Testing output data for free-running prediction.
        Shape: (output_size, test_size)

    u_train : np.ndarray
        Input signal portion used during training.
        Shape: (1, train_size)

    u_test : np.ndarray
        Input signal portion used during testing.
        Shape: (1, test_size)

    Yt : np.ndarray
        Target outputs for training regression (after warm-up).
        Shape: (output_size, train_size - warm_up)
    """

    # ------------------------------
    # Split output (system state) data
    # ------------------------------
    xs = data[0, :]
    ys = data[1, :]
    data_n = np.asarray([xs, ys])

    # ------------------------------
    # Split into training and testing portions
    # ------------------------------
    train_data = data_n[:, :train_size]
    test_data = data_n[:, train_size : train_size + test_size]

    # ------------------------------
    # Define target outputs for regression (skip warm-up)
    # ------------------------------
    Yt = data_n[:, warm_up + 1 : train_size + 1]

    # ------------------------------
    # Split input control signal and reshape
    # ------------------------------
    u_train = u[:train_size].reshape(1, -1)
    u_test = u[train_size : train_size + test_size].reshape(1, -1)

    # ------------------------------
    # Return all prepared data
    # ------------------------------
    return train_data, test_data, u_train, u_test, Yt
