import numpy as np
import random

class Linear():
    def __init__(self, in_features: int, out_features: int, activation, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.bias = bias
        self.reset_params()
    
    def __call__(self, x):
        # linear transformation through the layer

        _, w = x.shape

        if w == self.in_features:
            z = x @ self.A.transpose() + self.b
            self.x = x
            self.z = z
            self.a = self.activation(z)
            return self.a
        else:
            raise RuntimeError(f"Error: input cols ({w}) must be equal to in_features ({self.in_features})")
    
    def reset_params(self):
        """
        self.A = np.array([[random.uniform(-0.5, 0.5) for _ in range(self.in_features)] for _ in range(self.out_features)])
        self.b = np.array([0.0 for _ in range(self.out_features)])
        self.gradA = np.array([[0.0 for _ in range(self.in_features)] for _ in range(self.out_features)])
        self.gradb = np.array([0.0 for _ in range(self.out_features)])
        """

        fan_in = self.in_features   # input connections
        fan_out = self.out_features  # output connections

        # compute He standard deviation (good for relu activation) xavier -> sigmoid/tanh
        std_dev = np.sqrt(2.0 / fan_in)

        # initialize weights matrix
        self.A = np.random.randn(fan_out, fan_in) * std_dev
        self.b = np.zeros((1, fan_out))

        # seems to perform better with He initialization as opposed to uniform


def relu(x):

    """
    # really slow

    def func(x):
        return max(x, 0)
    
    vectorized_func = np.vectorize(func)
    result = vectorized_func(x)

    return result
    """

    # faster
    return np.maximum(x, 0)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)   # stability shift; result unchanged
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def deriv_relu(x):
    return x > 0

def one_hot_Y(y, num_classes=10):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

def mseLoss(a, y):
    #n = a.size # same as y
    loss = (a - y) ** 2
    return sum(np.sum(loss, axis=1)) / a.shape[0]

def cross_entropy(a, y):
    # a: softmax outputs (m, classes); y: one-hot targets (m, classes)
    m = a.shape[0]
    eps = 1e-12                          # prevent log(0)  -inf
    return -np.sum(y * np.log(a + eps)) / m