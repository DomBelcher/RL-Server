import torch
import torch.nn.functional as F
# import torch.optim as optim

from nn import RLModel

class Model():
  __instance = None
  def __init__(self, n_inputs, n_outputs):
    self.policy_model = RLModel(n_inputs, n_outputs)
    self.target_model = RLModel(n_inputs, n_outputs)
    self.target_model.load_state_dict(self.policy_model.state_dict())
    self.target_model.eval()

    self.train_X = torch.zeros(0, n_inputs)
    self.train_y = torch.zeros(0, n_outputs)

    self.n_inputs = n_inputs
    self.n_outputs = n_outputs

    # self.optimizer = torch.optim.RMSprop(self.policy_model.parameters())
    self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=0.001)

    Model.__instance = self

  def get_instance(n_inputs, n_outputs):
    if Model.__instance == None:
      print('######################')
      print('### creating model ###')
      print('######################')
      Model(n_inputs, n_outputs)
    return Model.__instance

  def get_action(self, state):
    with torch.no_grad():
      avs = self.target_model(state.unsqueeze(0))
      # print(avs)
      # print(avs.max(0)[1].view(1, 1))
      return avs.max(1)[1].view(1, 1)

  def optimise(self, batch):
    # print(batch.action)
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.stack(batch.next_state)

    state_action_values = self.policy_model(state_batch).gather(1, action_batch)
    next_state_values = self.target_model(next_state_batch).max(1)[0].detach()

    expected_state_action_values = (next_state_values * 0.99) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    self.optimizer.zero_grad()
    loss.backward()
    for param in self.policy_model.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()

  def to(self, device):
    self.target_model.to(device)
    self.policy_model.to(device)

# model_instance = None

# def get_model(n_inputs, n_outputs):
#   if model_instance is None:
#     print('######################')
#     print('### creating model ###')
#     print('######################')
#     model_instance = Model(n_inputs, n_outputs)

#   return model_instance
