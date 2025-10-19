# Lab 12 RNN
import torch
import torch.nn as nn

# CUDA设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

torch.manual_seed(777)  # reproducibility


idx2char = ['h', 'i', 'e', 'l', 'o']

# Teach hihell -> ihello
x_data = [[0, 1, 0, 2, 3, 3]]   # hihell
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3

y_data = [1, 0, 2, 3, 3, 4]    # ihello

# As we have one batch of samples, convert them to tensors
inputs = torch.tensor(x_one_hot, dtype=torch.float).to(device)  # shape: (1, 6, 5)
labels = torch.tensor(y_data, dtype=torch.long).to(device)      # shape: (6,)

num_classes = 5
input_size = 5  # one-hot size
hidden_size = 5  # output from the RNN. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 6  # |ihello| == 6
num_layers = 1  # one-layer rnn


class RNN(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)

    def forward(self, x):
        # Initialize hidden state
        # (num_layers * num_directions, batch, hidden_size) for batch_first=True
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Ensure correct shape (assign result)
        x = x.view(x.size(0), self.sequence_length, self.input_size)

        # Propagate input through RNN
        out, _ = self.rnn(x, h_0)
        return out.contiguous().view(-1, self.num_classes)


# Instantiate RNN model
rnn = RNN(num_classes, input_size, hidden_size, num_layers).to(device)
print(rnn)

# Set loss and optimizer function
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.1)

# Train the model
for epoch in range(100):
    outputs = rnn(inputs)            # outputs: (seq_len * batch, num_classes) -> (6, 5)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(1)
    idx = idx.detach().cpu().numpy()
    result_str = [idx2char[c] for c in idx]
    print(f"epoch: {epoch + 1}, loss: {loss.item():.3f}")
    print("Predicted string: ", ''.join(result_str))

print("Learning finished!")