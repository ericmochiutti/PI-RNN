import numpy as np


class ESN:
    """
    This implementation uses fixed recurrent reservoir weights and
    trains only the linear output layer via ridge regression.

    Parameters
    ----------
    input_size : int
        Number of input features.

    output_size : int
        Number of output features.

    train_size : int
        Number of time steps used for training.

    test_size : int
        Number of time steps used for testing.

    warm_up : int
        Number of initial steps discarded during training.

    random_seed : int
        Random seed for reproducibility.

    reservoir_size : int
        Number of reservoir neurons.
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
    ):
        np.random.seed(random_seed)

        self.input_size = input_size
        self.output_size = output_size
        self.train_size = train_size
        self.test_size = test_size
        self.warm_up = warm_up
        self.reservoir_size = reservoir_size

        # Weight matrices (initialized later)
        self.W = None
        self.Win = None
        self.Wf = None
        self.Wb = None
        self.Wout = None

        # Hyperparams
        self.leak_rate = None
        self.reg = None

        # Data placeholders
        self.input_data_train = None
        self.input_data_test = None
        self.output_data_train = None
        self.output_data_test = None
        self.Yt = None

        # Internal states
        self.X_train = None
        self.X_test = None
        self.last_state = None
        self.output_predict = None

    # ---------------------------------------------------------------
    # DATA ASSIGNMENT
    # ---------------------------------------------------------------
    def set_data(self, u_train, u_test, train_data, test_data, Yt):
        """
        Assign training and testing data from NumPy arrays.
        """
        self.input_data_train = np.asarray(u_train, dtype=np.float64)
        self.input_data_test = np.asarray(u_test, dtype=np.float64)
        self.output_data_train = np.asarray(train_data, dtype=np.float64)
        self.output_data_test = np.asarray(test_data, dtype=np.float64)
        self.Yt = np.asarray(Yt, dtype=np.float64)

    # ---------------------------------------------------------------
    # WEIGHT INITIALIZATION
    # ---------------------------------------------------------------
    def generate_weights(self, spectral_radius, Win_scaling, Wf_scaling, Wb_scaling):
        """
        Initialize all internal ESN weights.
        """
        # Recurrent weights
        W = np.random.uniform(-1, 1, (self.reservoir_size, self.reservoir_size))
        eigvals = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigvals))
        W *= spectral_radius / max_eig
        self.W = W

        # Input weights (ternary)
        self.Win = np.random.choice(
            [0, -Win_scaling, Win_scaling],
            size=(self.reservoir_size, self.input_size),
            p=[0.5, 0.25, 0.25],
        )

        # Feedback and bias
        self.Wf = np.random.uniform(
            -Wf_scaling, Wf_scaling, (self.reservoir_size, self.output_size)
        )
        self.Wb = np.random.uniform(-Wb_scaling, Wb_scaling, self.reservoir_size)

    # ---------------------------------------------------------------
    # TRAINING PHASE
    # ---------------------------------------------------------------
    def train_wout(self, leak_rate, reg):
        """
        Train the output weights (Wout) via ridge regression.
        """
        self.leak_rate = leak_rate
        self.reg = reg

        X = np.zeros((self.reservoir_size, self.train_size))
        x = np.zeros(self.reservoir_size)

        # Run reservoir (teacher forcing)
        for t in range(self.train_size):
            u_t = self.input_data_train[:, t]
            y_t = self.output_data_train[:, t]

            pre_activation = self.Win @ u_t + self.W @ x + self.Wf @ y_t + self.Wb
            x = (1 - leak_rate) * x + leak_rate * np.tanh(pre_activation)
            X[:, t] = x

        # Discard warm-up
        X_eff = X[:, self.warm_up :]
        Yt_eff = self.Yt

        # Ridge regression: Wout = Y Xᵀ (X Xᵀ + reg I)⁻¹
        A = X_eff @ X_eff.T + reg * np.eye(self.reservoir_size)
        B = Yt_eff @ X_eff.T
        self.Wout = np.linalg.solve(A.T, B.T).T

        self.X_train = X_eff
        self.last_state = x

    # ---------------------------------------------------------------
    # PREDICTION PHASE
    # ---------------------------------------------------------------
    def predict(self, Wout):
        """
        Run the ESN in free-running (testing) mode.
        """
        x = self.last_state.copy()
        y = self.output_data_train[:, -1].copy()

        Y = np.zeros((self.output_size, self.test_size))
        X_test = np.zeros((self.reservoir_size, self.test_size))

        for t in range(self.test_size):
            u_t = self.input_data_test[:, t]

            pre_activation = self.Win @ u_t + self.W @ x + self.Wf @ y + self.Wb
            x = (1 - self.leak_rate) * x + self.leak_rate * np.tanh(pre_activation)
            y = Wout @ x

            X_test[:, t] = x
            Y[:, t] = y

        self.X_test = X_test
        self.output_predict = Y
        return Y
