from flask_server import server as flask_server
from conn_interface import ConnectionInterface


interface = ConnectionInterface(n_inputs=7, n_actions=3)

flask_app = flask_server(interface)