import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ssl
import urllib.request
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from model import NN
from funcs import cross_entropy, one_hot_Y

csv = 'train.csv'
my_df = pd.read_csv(csv)


# Train Test Split!  Set X, y
X = pd.DataFrame(my_df.drop('label', axis=1))
y = my_df['label']

# Convert these to numpy arrays
X = X.values
y = y.values

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Convert X features to float tensors
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# Convert y labels to tensors long
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)


model = NN(784, 10, 128, 64)

# Train our model!
# Epochs? (one run thru all the training data in our network)
epochs = 50
losses = []
batch_size = 64

for i in range(epochs):
    perm = np.random.permutation(len(X_train))   # shuffle each epoch
    Xs, ys = X_train[perm], y_train[perm]
    for start in range(0, len(Xs), batch_size):
        xb = Xs[start:start+batch_size]
        yb = ys[start:start+batch_size]
        y_pred = model(xb)                 # forward (caches activations)
        model.back_prop(yb, 0.1)  # update on this batch
    """
    # Go forward and get a prediction
    y_pred = model(X_train) # Get predicted results
    model.back_prop(y_train, 0.1)
    """

    # Measure the loss/error, gonna be high at first
    loss = cross_entropy(y_pred, one_hot_Y(yb)) # predicted values vs the y_train

    # Keep Track of our losses
    losses.append(loss)

    # print every 10 epoch
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')



plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel('Epoch')
#plt.show()



# Evaluate Model on Test Data Set (validate model on test set)
with torch.no_grad():  # Basically turn off back propogation
    y_eval = model(X_test) # X_test are features from our test set, y_eval will be predictions
    loss = cross_entropy(y_eval, one_hot_Y(y_test)) # Find the loss or error

print(loss)


correct = 0

for i, data in enumerate(X_test):
    data = data.reshape(1,784)
    y_val = model(data)

    # Correct or not
    if y_val.argmax() == y_test[i]:
        correct += 1
    
    i += 1

print(f'Accuracy: {correct / y_test.size}')


#torch.save(model.state_dict(), 'digit_model.pt')

# needs to generalize more for better test performance