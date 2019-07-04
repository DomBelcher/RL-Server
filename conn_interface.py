import torch

from model import Model
from memory import ReplayMemory

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class ConnectionInterface():
  def __init__(self, n_inputs, n_actions, batch_size=128, train_frequency=10, memory_size=10000):
    self.model = Model.get_instance(n_inputs, n_actions)
    self.model.to(device)
    self.memory = ReplayMemory(memory_size)

    self.BATCH_SIZE = batch_size
    self.train_frequency = train_frequency

    self.tick = 0

  def get_action(self, s):
    state = torch.Tensor(s).to(device)
    action = self.model.get_action(state).item()

    return action

  def add_transition(self, s, a, r, ns):
    state = torch.Tensor(s).to(device)
    action = torch.LongTensor([[a]]).to(device)
    reward = torch.Tensor([r]).to(device)
    next_state = torch.Tensor(ns).to(device)

    self.memory.push(state, action, next_state, reward)

    if len(self.memory) >= self.BATCH_SIZE and self.tick % self.train_frequency == 0:
      print('Training')
      batch = self.memory.sample(self.BATCH_SIZE)
      self.model.optimise(batch)

    self.tick = self.tick + 1