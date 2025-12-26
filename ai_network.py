import torch
import torch.nn as nn
import numpy as np

class PongNet(nn.Module):
    """
    Neural network for Pong AI.
    Input: Game state (ball position, paddle position, velocities, etc.)
    Output: Continuous action in [-1, 1] where -1 = up, 1 = down, 0 = stay
    """
    def __init__(self, input_size=10, hidden_size=64, output_size=1):
        super(PongNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))  # Output in [-1, 1]
        return x
    
    def get_action(self, state):
        """
        Get continuous action from state.
        Returns: float in [-1, 1] where -1 = move up, 1 = move down, 0 = stay
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            output = self.forward(state_tensor)
            # Add small noise during training for exploration
            noise = torch.randn_like(output) * 0.1
            action = torch.clamp(output + noise, -1.0, 1.0)
            return action.item()
    
    def get_action_deterministic(self, state):
        """
        Get deterministic continuous action (no randomness).
        Returns: float in [-1, 1]
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            output = self.forward(state_tensor)
            return output.item()
    
    def get_weights(self):
        """Get all network weights as a flat numpy array"""
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)
    
    def set_weights(self, weights):
        """Set network weights from a flat numpy array"""
        idx = 0
        for param in self.parameters():
            param_size = param.data.numel()
            param_shape = param.data.shape
            param.data = torch.FloatTensor(
                weights[idx:idx + param_size].reshape(param_shape)
            )
            idx += param_size

