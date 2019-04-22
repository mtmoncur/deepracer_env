import numpy as np
import torch
from torch import nn

class ConvNetwork(nn.Module):
    def __init__(self, output_size):
        super(ConvNetwork, self).__init__()
        
        # convolutions to reduce 128x128x3 image down to output_size
        self._net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),  # -> [n, 32, 63, 63]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # -> [n, 64, 30, 30]
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2),  # -> [n, 64, 14, 14]
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),  # -> [n, 128, 6, 6]
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, stride=2),  # -> [n, 128, 2, 2]
        )
        self._fc = nn.Linear(128*2*2,output_size)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = self._net(x)
        out = out.reshape(-1, 128*2*2)
        out = self._fc(out)
        return out

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, action_size):
        super(MLP, self).__init__()
        layers = []
        
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, action_size))
        
        self._net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self._net(x)

class Actor(nn.Module):
    def __init__(self, net):
        super(Actor, self).__init__()
        self._net = net
        self._softmax = nn.Softmax(dim=1)

    def forward(self, x, get_action=True):
        scores = self._net(x)
        probs = self._softmax(scores)

        if not get_action:
            return probs
        
        batch_size = x.shape[0]
        actions = np.empty((batch_size, 1), dtype=np.uint8)
        probs_np = probs.cpu().detach().numpy()
        for i in range(batch_size):
            try:
                action_one_hot = np.random.multinomial(1, probs_np[i])
            except ValueError as e:
                probs_np = probs_np.astype(np.float64)
                probs_np /= np.sum(probs_np)
                action_one_hot = np.random.multinomial(1, probs_np[i])
            actions[i, 0] = np.argmax(action_one_hot)
        return probs, actions