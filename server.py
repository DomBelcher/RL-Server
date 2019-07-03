import logging

import torch
import numpy as np
from flask import Flask, request, jsonify
# from flask_restful import Resource, Api

from model import Model
from memory import ReplayMemory
# from server.memory import ReplayMemory


app = Flask(__name__)
# api = Api(app)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# class GetAction(Resource):
#   def get(self, state):

#     return { 'action': }

# class Transition(Resource):
#   def post(self, transition):

#     return

# api.add_resource(GetAction, '/action')
# api.add_resource(Transition, '/transition')
tick = 0

n_actions = 3
model = Model.get_instance(7, n_actions)
model.to(device)

memory = ReplayMemory(10000)
BATCH_SIZE = 128
train_frequency = 10
print(tick)

@app.route('/action', methods=['POST'])
def action():
  global device
  global model

  if request.method == 'POST':
    # print(request.is_json)
    content = request.get_json()
    # print(content)
    state = torch.Tensor(content['state']).to(device)
    action = model.get_action(state).item()

    return jsonify({ 'action': action })

@app.route('/transition', methods=['POST'])
def transition():
  if request.method ==  'POST':
    global tick
    global memory
    global BATCH_SIZE
    global train_frequency

    content = request.get_json()
    state = torch.Tensor(content['state']).to(device)
    action = torch.LongTensor([[content['action']]]).to(device)
    reward = torch.Tensor([content['reward']]).to(device)
    next_state = torch.Tensor(content['next_state']).to(device)

    memory.push(state, action, next_state, reward)

    if len(memory) >= BATCH_SIZE and tick % train_frequency == 0:
      print('Training')
      batch = memory.sample(BATCH_SIZE)
      model.optimise(batch)

    tick = tick + 1

    return jsonify({ 'status': 1 })

if __name__ == '__main__':
  host = '127.0.0.1'
  port = '3000'

  print('Server running on {}:{}'.format(host, port))
  app.run(host=host, port=port)