import torch
import torch.nn as nn

# CUDA设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# One hot encoding for each char in 'hello'
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

# One cell RNN input_dim (4) -> output_dim (2). sequence: 5
cell = nn.RNN(input_size=4, hidden_size=2, batch_first=True).to(device)

# (num_layers * num_directions, batch, hidden_size) whether batch_first=True or False
hidden = torch.randn(1, 1, 2).to(device)

# Propagate input through RNN
# Input: (batch, seq_len, input_size) when batch_first=True
print("Input each time step:")
inputs = torch.tensor([h, e, l, l, o], dtype=torch.float32).to(device)
for one in inputs:
    one = one.view(1, 1, -1)
    # Input: (batch, seq_len, input_size) when batch_first=True
    out, hidden = cell(one, hidden)
    print("one input size", one.size(), "out size", out.size())
print("=================================================")
print("")

print("Input whole sequence at once:")
# We can do the whole at once
# Propagate input through RNN
# Input: (batch, seq_len, input_size) when batch_first=True
inputs = inputs.view(1, 5, -1)
out, hidden = cell(inputs, hidden)
print("sequence input size", inputs.size(), "out size", out.size())
print("=================================================")
print("")

# hidden : (num_layers * num_directions, batch, hidden_size) whether batch_first=True or False
print("Input batch of sequences:")
hidden = torch.randn(1, 3, 2).to(device)

# One cell RNN input_dim (4) -> output_dim (2). sequence: 5, batch 3
# 3 batches 'hello', 'eolll', 'lleel'
# rank = (3, 5, 4)
inputs = torch.tensor([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=torch.float32).to(device)

# Propagate input through RNN
# Input: (batch, seq_len, input_size) when batch_first=True
# B x S x I
out, hidden = cell(inputs, hidden)
print("batch input size", inputs.size(), "out size", out.size())
print("=================================================")
print("")

# One cell RNN input_dim (4) -> output_dim (2)
print("Input batch of sequences with batch_first=False:")
cell = nn.RNN(input_size=4, hidden_size=2).to(device)

# The given dimensions dim0 and dim1 are swapped.
inputs = inputs.transpose(dim0=0, dim1=1)
# Propagate input through RNN
# Input: (seq_len, batch_size, input_size) when batch_first=False (default)
# S x B x I
out, hidden = cell(inputs, hidden)
print("batch input size", inputs.size(), "out size", out.size())
print("=================================================")
