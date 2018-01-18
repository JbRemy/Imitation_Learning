import numpy as np
import gym

import tensorflow as tf
import parameters
from network import Neural_Network
from utils import Fetch_trajectories
from agent import Agent
import time

class DAGGER(object):

    def __init__(self, game):

        self.game = game
        self.parameters = getattr(parameters, game)

        self.env = gym.make(self.parameters["env_name"])
        self.Network = Neural_Network(game)
        self.path = self.parameters['path']
        self.agent = Agent(self.env, self.path, self.Network)
        self.beta = self.parameters['beta']
        self.device = self.parameters['device']


    def train(self, n_iterations):

        writer = tf.summary.FileWriter('{}/Model/logs/train/'.format(self.path))

        for lap in range(n_iterations):
            with open('{}/list.txt', 'r') as file:
                data_set_size = len(file.readlines())
            Fetch_trajectories(self.agent, beta=self.beta**lap)
            self.Neural_Network.fit(self.device, self.path, '{}/Model_{}'.format(self.path, lap), writer, time.time(), lap)

        writer.close()



