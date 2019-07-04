import logging

import torch
import numpy as np
from flask import Flask, request, jsonify

from model import Model
from memory import ReplayMemory

def server(interface):
  app = Flask(__name__)

  log = logging.getLogger('werkzeug')
  log.setLevel(logging.ERROR)

  @app.route('/action', methods=['POST'])
  def action():
    if request.method == 'POST':
      content = request.get_json()
      state = content['state']
      action = interface.get_action(state)

      return jsonify({ 'action': action })

  @app.route('/transition', methods=['POST'])
  def transition():
    if request.method ==  'POST':
      content = request.get_json()
      state = content['state']
      action = content['action']
      reward = content['reward']
      next_state = content['next_state']

      interface.add_transition(state, action, reward, next_state)

      return jsonify({ 'status': 1 })

  host = '127.0.0.1'
  port = '3000'

  print('Server running on {}:{}'.format(host, port))
  app.run(host=host, port=port)

  return app