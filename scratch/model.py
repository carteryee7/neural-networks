from funcs import Linear, relu, softmax

class NN:
    def __init__(self, in_features=11, out_features=4, h1=64, h2=32):
        self.fc1 = Linear(in_features, h1)
        self.fc2 = Linear(h1, h2)
        self.out = Linear(h2, out_features)
    
    def __call__(self, x):  # forward prop
        return self.forward(x)
    
    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        output = softmax(self.out(x))

        return output
    
    def back_prop(self):
        