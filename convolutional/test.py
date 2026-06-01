from model import model
from drawer import row
import torch

# Load the Saved Model
new_model = model
new_model.load_state_dict(torch.load('cnn_model2.pt'))

# Make sure it loaded correctly
new_model.eval()

new_digit = torch.tensor(row, dtype=torch.float32) # 2D
new_digit = new_digit.view(-1, 1, 28, 28) # 4D
print(new_digit.shape)

with torch.no_grad():
    pred = new_model(new_digit) # same output as "new_model.forward(new_digit)" roughly
    print(pred)
    print(f"\n{pred.argmax().item()}")