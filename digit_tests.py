from digit_model import Model
from digit_drawer import row
import torch

# Load the Saved Model
new_model = Model()
new_model.load_state_dict(torch.load('digit_model.pt'))

# Make sure it loaded correctly
new_model.eval()

new_digit = torch.tensor(row, dtype=torch.float32)

with torch.no_grad():
    pred = new_model(new_digit)
    print(pred)
    print(f"\n{pred.argmax().item()}")

