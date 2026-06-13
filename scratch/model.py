from funcs import Linear, relu, softmax, deriv_relu, one_hot_Y

class NN:
    def __init__(self, in_features=11, out_features=4, h1=64, h2=32):
        self.fc1 = Linear(in_features, h1, relu)
        self.fc2 = Linear(h1, h2, relu)
        self.out = Linear(h2, out_features, softmax)
    
    def __call__(self, x):  # forward prop
        return self.forward(x)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.out(x)

        return output
    
    def back_prop(self, y, alpha):
        y = one_hot_Y(y, self.out.out_features)
        m = y.shape[0]

        dz = self.out.a - y

        layers = [(self.out, self.fc2.z), (self.fc2, self.fc1.z), (self.fc1, self.fc1.x)]
        
        for layer, prev_z in layers:
            gradA = dz.transpose() @ layer.x / m
            gradb = dz.mean(axis=0)
            da = dz @ layer.A
            dz = da * deriv_relu(prev_z)
            layer.A = layer.A - (alpha * gradA)
            layer.b = layer.b - (alpha * gradb)