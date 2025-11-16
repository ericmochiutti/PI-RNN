import os
from copy import deepcopy as deepcopy

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import tensorflow_probability as tfp


class PIESN:
    def __init__(self, ESN, dt, cons_phy, colocation_points, subsampling):
        self.ESN = ESN
        self.PI_ESN = deepcopy(ESN)
        self.dt = dt
        self.cons_phy = cons_phy
        self.colocation_points = colocation_points
        self.subsampling = subsampling
        self.loss_data_list = []
        self.loss_physics_list = []
        self.output_prediction_PI = None
        self.MSE_list = []
        self.param_data_list = []
        self.param_phy_list = []

    def physical_regularization_vanderpol(self):
        def Wout_to_woutparam(Wout):
            # TRANSFORM THE 2D Wout array into 1D array for LBFGS FUNCTION
            Wout_1 = np.reshape(Wout[0, :], self.PI_ESN.reservoir_size)
            Wout_2 = np.reshape(Wout[1, :], self.PI_ESN.reservoir_size)
            Wout_param = np.hstack((Wout_1, Wout_2))  # initparam[0], initparam[1]))

            return Wout_param

        def woutparam_to_Wout(Wout_param):
            # TRANSFORM THE 1D Wout array into 2D array
            Wout = np.reshape(
                Wout_param[: int(self.PI_ESN.reservoir_size * self.PI_ESN.output_size)],
                (self.PI_ESN.output_size, self.PI_ESN.reservoir_size),
            )

            return Wout

        def loss_training_data(Wout_param):
            y1 = tf.linalg.matvec(
                self.PI_ESN.X_train.T, Wout_param[: self.PI_ESN.reservoir_size]
            )
            y2 = tf.linalg.matvec(
                self.PI_ESN.X_train.T,
                Wout_param[
                    self.PI_ESN.reservoir_size : int(self.PI_ESN.reservoir_size * 2)
                ],
            )

            loss_data_1 = (1 / (self.PI_ESN.train_size)) * tf.experimental.numpy.sum(
                (y1 - self.PI_ESN.Yt[0, :]) ** 2
            )
            loss_data_2 = (1 / (self.PI_ESN.train_size)) * tf.experimental.numpy.sum(
                (y2 - self.PI_ESN.Yt[1, :]) ** 2
            )
            loss_data = (loss_data_1 + loss_data_2) / 2

            return loss_data

        def loss_collocation_points_physics(Wout_param):
            u_data_phy = self.PI_ESN.input_data_test[
                :,
                : self.PI_ESN.test_size - 1,
            ]

            dt_phy = self.dt * self.subsampling

            y1 = tf.linalg.matvec(
                self.PI_ESN.X_test.T, Wout_param[: self.PI_ESN.reservoir_size]
            )
            y2 = tf.linalg.matvec(
                self.PI_ESN.X_test.T,
                Wout_param[
                    self.PI_ESN.reservoir_size : int(self.PI_ESN.reservoir_size * 2)
                ],
            )

            h1 = y1[: self.colocation_points - 1]
            h2 = y2[: self.colocation_points - 1]

            hn1 = y1[1 : self.colocation_points]
            hn2 = y2[1 : self.colocation_points]

            mu = 1

            loss_physics_1 = (h2 * dt_phy - hn1 + h1) ** 2
            loss_physics_2 = (
                (
                    -mu * (h1**2 - 1) * h2
                    - h1
                    + u_data_phy[0, : (self.colocation_points - 1)]
                )
                * dt_phy
                + h2
                - hn2
            ) ** 2

            loss_physics = tf.experimental.numpy.sum(
                loss_physics_1 + loss_physics_2
            ) / (self.PI_ESN.output_size * self.colocation_points)

            return loss_physics

        def loss_gradient(Wout_param):
            Wout_param = tf.Variable(Wout_param, trainable=True)

            with tf.GradientTape() as tape:
                loss_data = loss_training_data(Wout_param)
                self.loss_data_list.append(loss_data)
                loss_physics = loss_collocation_points_physics(Wout_param)
                self.loss_physics_list.append(loss_physics)

                # self adaptive loss
                # lambda1 = (1 / 2) * tf.math.exp(-Wout_param[-1])
                # lambda2 = (1 / 2) * tf.math.exp(-Wout_param[-2])

                cost = (
                    1 * loss_data + self.cons_phy * loss_physics
                    # + Wout_param[-1]
                    # + Wout_param[-2]
                )

                dcost = tape.gradient(cost, Wout_param)

                return cost, dcost

        def np_value(tensor):
            if isinstance(tensor, tuple):
                return type(tensor)(*(np_value(t) for t in tensor))
            else:
                return tensor.numpy()

        def run(optimizer):
            optimizer()  # Warmup.

            result = optimizer()

            return np_value(result)

        def cost_gradient_lbfgs():
            # init_param = [1, 1]
            init_value = Wout_to_woutparam(self.PI_ESN.Wout)  # ,init_param

            return tfp.optimizer.lbfgs_minimize(
                loss_gradient,
                initial_position=init_value,
                tolerance=1e-9,
                max_iterations=100,
                max_line_search_iterations=100,
            )

        for number_X_test_update in range(100):
            results = run(cost_gradient_lbfgs)
            theta = woutparam_to_Wout(results.position)
            self.PI_ESN.Wout = theta
            self.PI_ESN.predict(theta)

            MSE_1 = np.sum(
                ((theta @ self.PI_ESN.X_test)[0, :] - self.ESN.output_data_test[0, :])
                ** 2
            ) / (self.PI_ESN.test_size)
            MSE_2 = np.sum(
                ((theta @ self.PI_ESN.X_test)[1, :] - self.ESN.output_data_test[1, :])
                ** 2
            ) / (self.PI_ESN.test_size)
            MSE = (MSE_1 + MSE_2) / 2

            MSE_ESN_1 = np.sum(
                (
                    (self.ESN.Wout @ self.ESN.X_test)[0, :]
                    - self.ESN.output_data_test[0, :]
                )
                ** 2
            ) / (self.PI_ESN.test_size)
            MSE_ESN_2 = np.sum(
                (
                    (self.ESN.Wout @ self.ESN.X_test)[1, :]
                    - self.ESN.output_data_test[1, :]
                )
                ** 2
            ) / (self.PI_ESN.test_size)
            MSE_ESN = (MSE_ESN_1 + MSE_ESN_2) / 2

            self.MSE_list.append(MSE)
            print(
                "Epoch:",
                number_X_test_update,
                " error PI-ESN:",
                MSE,
                " error ESN:",
                MSE_ESN,
            )

        self.output_prediction_PI = theta @ self.PI_ESN.X_test

        return MSE
