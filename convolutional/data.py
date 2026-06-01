import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from model import model

csv = 'train.csv'
df = pd.read_csv(csv)

"""
def func(x):
    if x > 20:
        return 1
    else:
        return 0

vectorized_func = np.vectorize(func)
x = pd.DataFrame(vectorized_func(my_df.drop('label', axis=1)))
y = my_df['label']
"""

# Train Test Split!  Set X, y
x = df.drop('label', axis=1)
y = df['label']

# Normalize pixel values to [0, 1] range
x = x / 255.0

# Convert these to numpy arrays
X = x.values
y = y.values

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Reshape to [batch, channels, height, width] for conv2d
X_train = X_train.view(-1, 1, 28, 28)
X_test = X_test.view(-1, 1, 28, 28)

# Convert y labels to tensors long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Use GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
model = model.to(device)


# Set the criterion of model to measure the error, how far off the predictions are from the data
criterion = nn.CrossEntropyLoss()
# Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (epochs), lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


# Train our model!
# Epochs? (one run thru all the training data in our network)
epochs = 100 # try with 40
batch_size = 32
losses = []

# Create batches
num_batches = len(X_train) // batch_size

for i in range(epochs + 1):
    epoch_loss = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        
        # Go forward and get a prediction
        y_pred = model(X_batch)
        
        # Measure the loss/error
        loss = criterion(y_pred, y_batch)
        epoch_loss += loss.item()
        
        # Do some back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_epoch_loss = epoch_loss / num_batches
    losses.append(avg_epoch_loss)
    
    # print every 10 epoch
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {avg_epoch_loss:.4f}')

"""
plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel('Epoch')
#plt.show()
"""


# Evaluate Model on Test Data Set (validate model on test set)
with torch.no_grad():  # Basically turn off back propogation
    y_eval = model(X_test) # X_test are features from our test set, y_eval will be predictions
    loss = criterion(y_eval, y_test) # Find the loss or error

print(loss)

torch.save(model.state_dict(), 'cnn_model.pt')