import numpy as np
import torch


class ESN(torch.nn.Module):
    """
    Echo State Network (ESN) implemented using PyTorch tensors.

    This implementation supports NumPy input data for compatibility with
    scientific workflows, but internally uses PyTorch for numerical operations
    (CPU or GPU). The model consists of a fixed recurrent reservoir and a
    trainable linear output layer.

    Parameters
    ----------
    input_size : int
        Number of input features (e.g., 1 for a scalar control input).

    output_size : int
        Number of output features (e.g., 2 for x(t) and y(t)).

    train_size : int
        Number of time steps used for training.

    test_size : int
        Number of time steps used for testing (free-running phase).

    warm_up : int
        Number of initial time steps to discard (reservoir stabilization period).

    random_seed : int
        Random seed for reproducibility.

    reservoir_size : int
        Number of neurons in the reservoir.

    device : torch.device or str, optional
        Device where tensors are stored ("cpu" or "cuda"). Defaults to auto-detect.
    """

    def __init__(
        self,
        input_size,
        output_size,
        train_size,
        test_size,
        warm_up,
        random_seed,
        reservoir_size,
        device=None,
    ):
        super().__init__()
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.input_size = input_size
        self.output_size = output_size
        self.train_size = train_size
        self.test_size = test_size
        self.warm_up = warm_up
        self.reservoir_size = reservoir_size

        # Select device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Weight matrices (to be initialized later)
        self.W = None  # Recurrent weights
        self.Win = None  # Input weights
        self.Wf = None  # Feedback weights
        self.Wb = None  # Bias weights
        self.Wout = None  # Output (trainable) weights

        # Training hyperparameters
        self.leak_rate = None
        self.reg = None

        # States and data placeholders
        self.X_train = None
        self.X_test = None
        self.last_state = None
        self.output_predict = None

        # Data placeholders (assigned using `set_data`)
        self.input_data_train = None
        self.input_data_test = None
        self.output_data_train = None
        self.output_data_test = None
        self.Yt = None  # target outputs

    # ---------------------------------------------------------------
    # DATA ASSIGNMENT
    # ---------------------------------------------------------------
    def set_data(self, u_train, u_test, train_data, test_data, Yt):
        """
        Assign training and testing data from NumPy arrays.

        Parameters
        ----------
        u_train : np.ndarray
            Input data used during training, shape (1, train_size).

        u_test : np.ndarray
            Input data used during testing, shape (1, test_size).

        train_data : np.ndarray
            Output data for training (teacher-forced), shape (output_size, train_size).

        test_data : np.ndarray
            Output data for testing (ground truth), shape (output_size, test_size).

        Yt : np.ndarray
            Target outputs for training after warm-up, shape (output_size, train_size - warm_up).
        """
        # Convert all numpy arrays to torch tensors
        self.input_data_train = torch.tensor(
            u_train, dtype=torch.float32, device=self.device
        )
        self.input_data_test = torch.tensor(
            u_test, dtype=torch.float32, device=self.device
        )
        self.output_data_train = torch.tensor(
            train_data, dtype=torch.float32, device=self.device
        )
        self.output_data_test = torch.tensor(
            test_data, dtype=torch.float32, device=self.device
        )
        self.Yt = torch.tensor(Yt, dtype=torch.float32, device=self.device)

    # ---------------------------------------------------------------
    # WEIGHT INITIALIZATION
    # ---------------------------------------------------------------
    def generate_weights(self, spectral_radius, Win_scaling, Wf_scaling, Wb_scaling):
        """
        Initialize and scale all internal ESN weights.

        Parameters
        ----------
        spectral_radius : float
            Desired spectral radius for the internal recurrent matrix.

        Win_scaling : float
            Scaling factor for input weights.

        Wf_scaling : float
            Scaling factor for feedback weights.

        Wb_scaling : float
            Scaling factor for bias weights.
        """
        # Recurrent reservoir weights
        W = np.random.uniform(-1, 1, (self.reservoir_size, self.reservoir_size))
        eigvals = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigvals))
        W *= spectral_radius / max_eig
        self.W = torch.tensor(W, dtype=torch.float32, device=self.device)

        # Input weights (sparse ternary initialization)
        Win = np.random.choice(
            [0, -Win_scaling, Win_scaling],
            size=(self.reservoir_size, self.input_size),
            p=[0.5, 0.25, 0.25],
        )
        self.Win = torch.tensor(Win, dtype=torch.float32, device=self.device)

        # Feedback and bias weights
        self.Wf = torch.empty(
            (self.reservoir_size, self.output_size), device=self.device
        ).uniform_(-Wf_scaling, Wf_scaling)
        self.Wb = torch.empty((self.reservoir_size), device=self.device).uniform_(
            -Wb_scaling, Wb_scaling
        )

    # ---------------------------------------------------------------
    # TRAINING PHASE
    # ---------------------------------------------------------------
    def train_wout(self, leak_rate, reg):
        """
        Train the ESN output weights (Wout) using ridge regression.

        Parameters
        ----------
        leak_rate : float
            Leaking rate controlling how much of the new state replaces the old one.

        reg : float
            Regularization factor for ridge regression.
        """
        self.leak_rate = leak_rate
        self.reg = reg

        X = torch.zeros((self.reservoir_size, self.train_size), device=self.device)
        x = torch.zeros((self.reservoir_size), device=self.device)

        # Run the reservoir during training (teacher-forced)
        for t in range(self.train_size):
            u_t = self.input_data_train[:, t]
            y_t = self.output_data_train[:, t]

            # Reservoir update equation
            pre_activation = self.Win @ u_t + self.W @ x + self.Wf @ y_t + self.Wb
            x = (1 - leak_rate) * x + leak_rate * torch.tanh(pre_activation)
            X[:, t] = x

        # Discard the warm-up period
        X_eff = X[:, self.warm_up :]
        Yt_eff = self.Yt

        # Ridge regression
        identity = torch.eye(self.reservoir_size, device=self.device)
        A = X_eff @ X_eff.T + reg * identity
        B = Yt_eff @ X_eff.T
        self.Wout = torch.linalg.solve(A.T, B.T).T

        self.X_train = X_eff
        self.last_state = x

    # ---------------------------------------------------------------
    # PREDICTION PHASE
    # ---------------------------------------------------------------
    def predict(self, Wout):
        """
        Run the Echo State Network (ESN) in free-running mode (testing phase).

        During this phase, the ESN evolves autonomously — its future outputs are
        generated using the trained output weights (Wout) and its own previous
        predictions, instead of the true target outputs.

        Parameters
        ----------
        Wout : torch.Tensor
            Trained output weight matrix of shape (output_size, reservoir_size),
            typically obtained from the training phase.

        Returns
        -------
        Y : np.ndarray
            Predicted system outputs for the testing phase.
            Shape: (output_size, test_size)
        """

        # Start from the last known reservoir state after training
        x = self.last_state.clone()

        # Initialize with the last true training output (teacher-forced last step)
        y = self.output_data_train[:, -1]

        # Allocate memory for predictions and reservoir states
        Y = torch.zeros((self.output_size, self.test_size), device=self.device)
        X_test = torch.zeros((self.reservoir_size, self.test_size), device=self.device)

        # --- Main testing loop ---
        for t in range(self.test_size):
            # Current input (control signal)
            u_t = self.input_data_test[:, t]

            # Update the internal reservoir state
            pre_activation = self.Win @ u_t + self.W @ x + self.Wf @ y + self.Wb
            x = (1 - self.leak_rate) * x + self.leak_rate * torch.tanh(pre_activation)
            # Generate the network's output using the provided Wout
            y = Wout @ x

            # Store state and output
            X_test[:, t] = x
            Y[:, t] = y

        self.X_test = X_test
        self.output_predict = Y.detach().cpu().numpy()

        return self.output_predict
