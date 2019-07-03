from torch import nn
import torch.nn.functional as F

hidden_dim = 128

class RLModel(nn.Module):
  def __init__(self, n_inputs, n_outputs):
    super(RLModel, self).__init__()
    self.fc1 = nn.Linear(n_inputs, hidden_dim)
    self.bn1 = nn.BatchNorm1d(hidden_dim)

    self.fc1a = nn.Linear(hidden_dim, hidden_dim)
    self.bn1a = nn.BatchNorm1d(hidden_dim)

    self.fc2 = nn.Linear(hidden_dim, n_outputs)
    self.bn2 = nn.BatchNorm1d(n_outputs)

    self.n_outputs = n_outputs

  def forward(self, x):
    out = self.fc1(x)
    out = self.bn1(out)
    out = F.relu(out)

    out = self.fc1a(out)
    out = self.bn1a(out)
    out = F.relu(out)

    out = self.fc2(out)
    out = self.bn2(out)
    # out = F.relu(out)
    
    # out = out.view(self.n_outputs, -1)
    # print(out.shape)

    return out