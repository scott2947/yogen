import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, device, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=32, batch_first=True)
        self.do = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(32, 16)
        self.bn = nn.BatchNorm1d(16)
        self.act1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(16, 1)
        self.act2 = nn.Sigmoid()
    
    def forward(self, x, states):
        out, states = self.lstm(x, states) # LSTM layer
        out = out[:, -1, :] # Select the last timestep only
        out = self.do(out) # Apply dropout
        out = self.act1(self.bn(self.fc1(out))) # First dense layer
        out = self.act2(self.fc2(out)) # Second dense layer
        return out, states
    
    def init_state(self, batch_size):
        hidden = torch.zeros(1, batch_size, 32, device=self.device)
        cell = torch.zeros(1, batch_size, 32, device=self.device)
        return hidden, cell