import numpy as np
import tensorflow as tf


class Agent(object):

    def __init__(self, env, data_path, network, device='/CPU:0'):
        '''

        :param env:
        :param data_path:
        :param network_path:
        :param device:
        '''
        self.env = env
        self.data_path = data_path
        self.device = device
        self.Network = Network

    def policy(self, X):
        '''
        chooses the action to execute
        :param state:
        :return:
        '''

        return self.Network.predict(X)


