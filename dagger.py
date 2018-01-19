import numpy as np
import gym
import matplotlib.pyplot as plt
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
            Fetch_trajectories(self.agent, beta=self.beta**lap)
            self.Network.fit(self.device, self.path, '{}/Model_{}'.format(self.path, lap), writer, time.time(), lap)
        writer.close()


if __name__ == '__main__':
    '''
    The following was used to locally update the data set, and then train each iterations on a remote Azure VM
    '''
    lap = 1

    net = Neural_Network('CarRacing', network_path='{0}/Model_{1}'.format('CarRacing', lap-1)) # if lap > 0
    agent = Agent("CarRacing", "pong", net)

    agent.play(lap=lap-1)
    Fetch_trajectories(agent, lap-1, beta=0.6**lap)

    writer = tf.summary.FileWriter('{0}/Model_{1}/logs/train/'.format('pong', lap))
    net = Neural_Network('CarRacing')
    net.fit('/GPU:0', Data_path='pong', save_path="pong/Model_{}".format(lap), writer=writer, start_time=time.time(), lap=lap)
    writer.close()
