import torch
import pdb

# CUDA设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = torch.tensor([1.0], requires_grad=True).to(device)

# our model forward pass
def forward(x):
    return x * w

# Loss function
def loss(y_pred, y_val):
    return (y_pred - y_val) ** 2

# Before training
print("Prediction (before training)",  4, forward(4).item())

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        # 将数据移动到设备
        x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
        
        y_pred = forward(x_val) # 1) Forward pass
        l = loss(y_pred, y_val) # 2) Compute loss
        l.backward() # 3) Back propagation to update weights
        
        # 检查梯度是否存在
        if w.grad is not None:
            print("\tgrad: ", x_val.item(), y_val.item(), w.grad.item())
            w.data = w.data - 0.01 * w.grad.item()
            # Manually zero the gradients after updating weights
            w.grad.zero_()
        else:
            print("\tgrad: ", x_val.item(), y_val.item(), "No gradient")

    print(f"Epoch: {epoch} | Loss: {l.item()}")

# After training
print("Prediction (after training)",  4, forward(4).item())
