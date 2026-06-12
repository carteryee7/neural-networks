import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ssl
import urllib.request
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from model import NN
from funcs import mseLoss

"""
url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
context = ssl._create_unverified_context()
with urllib.request.urlopen(url, context=context) as response:
    my_df = pd.read_csv(response)

my_df['variety'] = my_df['variety'].map({
    'Setosa': 0,
    'Versicolor': 1,
    'Virginica': 2,
}).astype('int64')
"""
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
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert y labels to tensors long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


model = NN(784, 10, 128, 64)

# Train our model!
# Epochs? (one run thru all the training data in our network)
epochs = 100
losses = []
for i in range(epochs):
    # Go forward and get a prediction
    y_pred = model(X_train) # Get predicted results
    model.back_prop(y_train, 0.01)

    # Measure the loss/error, gonna be high at first
    loss = mseLoss(y_pred, y_train) # predicted values vs the y_train

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
    loss = mseLoss(y_eval, y_test) # Find the loss or error

print(loss)

"""
correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        # Will tell us what type of flower class our network thinks it is
        print(f'{i+1}.)  {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')

        # Correct or not
        if y_val.argmax().item() == y_test[i]:
            correct +=1

print(f'We got {correct} correct!')
"""

#torch.save(model.state_dict(), 'digit_model.pt')

