import torch
import torch.nn as nn

class EEG_LSTM(nn.Module):
    def __init__(self, input_size=14, hidden_size=64, num_layers=1):
        super(EEG_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.a = nn.ReLU6()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)  # Regression output

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, (hn, _) = self.lstm(x)  # Take the final hidden state
        out = self.a(hn[-1])  # (batch, hidden_size)
        out = self.a(self.fc1(out))
        out = self.a(self.fc2(out)) # (batch, 1)
        out = self.output(out)
        return out.squeeze(1)  # (batch,)
