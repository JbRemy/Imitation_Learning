import numpy as np
import tensorflow as tf
import gym
import network

import parameters

class Agent(object):

    def __init__(self, env, data_path, Network, device='/CPU:0'):
        '''

        :param env:
        :param data_path:
        :param network_path:
        :param device:
        '''
        self.parameters = getattr(parameters, env)
        self.env = gym.make(self.parameters["env_name"])
        self.data_path = data_path
        self.device = device
        self.Network = Network
        self.n_actions = self.parameters['nb_actions']

    def policy(self, X):
        '''
        chooses the action to execute
        :param state:
        :return:
        '''

        return self.Network.predict(X)

