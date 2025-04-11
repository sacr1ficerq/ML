from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function
        self.delta = 1.0

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        error = y - x @ self.w
        l = x.shape[0]
        # match self.loss_function:
        #     case LossFunction.MSE:
        #         return error.T @ error / l
        #     case LossFunction.LogCosh:
        #         return np.mean(np.log(np.cosh(-error)))
        #     case LossFunction.MAE:
        #         return np.mean(np.abs(error))
        #     case LossFunction.Huber:
        #         q_loss = (error**2) / 2
        #         abs_loss = self.delta * (np.abs(error) - self.delta / 2)
        #         loss = np.where(np.abs(error) <= self.delta, q_loss, abs_loss)
        #         return np.mean(loss)


        # Это мне написал chat GPT, потому что мне лень переписывать верхний код без match 
        loss_functions = {
            LossFunction.MSE: lambda error, l: 1/l * (error.T @ error),
            LossFunction.LogCosh: lambda error, l: 1/l*(np.log(np.cosh(-error))),
            LossFunction.MAE: lambda error, l: 1/l*(np.abs(error)),
            LossFunction.Huber: lambda error, l: 1/l(
                np.where(
                    np.abs(error) <= self.delta,
                    (error**2) / 2,
                    self.delta * (np.abs(error) - self.delta / 2)
                )
            )
        }

        return loss_functions[self.loss_function](error, l)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        return x @ self.w


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """
    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights with respect to gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        lr = self.lr()
        self.w -= lr * gradient
        return -lr * gradient

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        error =  y - x @ self.w
        l = x.shape[0]

        # match self.loss_function:
        #     case LossFunction.MSE:
        #         return -2/l * x.T @ error
        #     case LossFunction.LogCosh:
        #         return - x.T @ np.tanh(error)
        #     case LossFunction.MAE:
        #         return -1/l * x.T @ np.sign(error)
        #     case LossFunction.Huber:
        #         l = np.abs(error) <= self.delta
        #         m = np.abs(error) > self.delta
        #         less = x[l].T @ error[l]
        #         more = x[m].T @ np.sign(error[m])
        #         return -1/l * (less + self.delta * more)

        # Это мне написал chat GPT, потому что мне лень переписывать верхний код без match 
        gradient_functions = {
            LossFunction.MSE: lambda x, error, l: \
                (-2/l * x.T) @ error,
            LossFunction.LogCosh: lambda x, error, l: \
                (-1/l * x.T) @ np.tanh(error),
            LossFunction.MAE: lambda x, error, l: \
                (-1/l * x.T) @ np.sign(error),
            LossFunction.Huber: lambda x, error, l: \
                -1/l * (
                    x[np.abs(error) <= self.delta].T @ error[np.abs(error) <= self.delta] +
                    self.delta * x[np.abs(error) > self.delta].T @ np.sign(error[np.abs(error) > self.delta])
                )
        }

        return gradient_functions[self.loss_function](x, error, l)

 


class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """
    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size


    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        l = x.shape[0]
        ind = np.random.randint(low=0, high=l - 1, size=self.batch_size)
        x = x[ind].copy()
        y = y[ind].copy()

        error =  y - x @ self.w

        # match self.loss_function:
        #     case LossFunction.MSE:
        #         return -2/l * x.T @ error
        #     case LossFunction.LogCosh:
        #         return - x.T @ np.tanh(error)
        #     case LossFunction.MAE:
        #         return -1/l * x.T @ np.sign(error)
        #     case LossFunction.Huber:
        #         l = np.abs(error) <= self.delta
        #         m = np.abs(error) > self.delta
        #         less = x[l].T @ error[l]
        #         more = x[m].T @ np.sign(error[m])
        #         return -1/l * (less + self.delta * more)
        
        # Это мне написал chat GPT, потому что мне лень переписывать верхний код без match 
        gradient_functions = {
            LossFunction.MSE: lambda x, error, l: \
                (-2/l * x.T) @ error,
            LossFunction.LogCosh: lambda x, error, l: \
                (-1/l * x.T) @ np.tanh(error),
            LossFunction.MAE: lambda x, error, l: \
                (-1/l * x.T) @ np.sign(error),
            LossFunction.Huber: lambda x, error, l: \
                -1/l * (
                    x[np.abs(error) <= self.delta].T @ error[np.abs(error) <= self.delta] +
                    self.delta * x[np.abs(error) > self.delta].T @ np.sign(error[np.abs(error) > self.delta])
                )
        }

        return gradient_functions[self.loss_function](x, error, l)


class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)


    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights with respect to gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """

        lr = self.lr()
        self.h = self.alpha * self.h + lr * gradient
        self.w -= self.h
        return -self.h


class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights & params
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        prev = self.w.copy()
        lr = self.lr()

        self.iteration += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * gradient ** 2

        m_hat = self.m / (1 - self.beta_1 ** self.iteration)
        v_hat = self.v / (1 - self.beta_2 ** self.iteration)

        self.w -= lr / (np.sqrt(v_hat) + self.eps) * m_hat

        return self.w - prev



class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient: np.ndarray = np.zeros_like(x.shape[1])  # TODO: replace with L2 gradient calculation

        return super().calc_gradient(x, y) + l2_gradient * self.mu


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
