import numpy as np
import random

class Linear():
    def __init__(self, in_features: int, out_features: int, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.reset_params()
    
    def __call__(self, x):
        h, w = x.shape

        if w == self.in_features:
            z = x @ self.A.transpose() + self.b
            return z
        else:
            print(f"Error: input cols ({w}) must be equal to in_features ({self.in_features})")
    
    def reset_params(self):
        self.A = np.array([[random.uniform(-0.5, 0.5) for _ in range(self.in_features)] for _ in range(self.out_features)])
        self.b = np.array([0.0 for _ in range(self.out_features)])


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
    x = np.exp(x) # convert each element to e^x
    summation = sum(x)
    out = x / summation
    
    return out

