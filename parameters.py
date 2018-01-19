import tensorflow as tf

pong = {"env_name": 'Pong-v0',

        'n_epochs': 10,
        "batch_size": 6,
        "optimizer": tf.train.AdamOptimizer,
        'learning_rate': 1e-3,
        'beta' : 0.9,
        'device' : '/CPU:0',

        'path': '/pong',

        "nb_agents": 1,
        'n_actions': 2,

        "input_W": 210,
        'input_H': 160,
        'input_C': 3,

        'n_hidden_layers': 1,
        'n_hidden_layers_nodes': 32
        }
