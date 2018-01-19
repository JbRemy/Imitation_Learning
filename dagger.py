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
            self.Network.fit(self.device, self.path, '{}/Model_{}'.format(self.path, lap), writer, time.time(), lap)
        writer.close()


if __name__ == '__main__':
    '''
    The following was used to locally update the data set, and then train each iterations on a remote Azure VM
    '''
    # to exec in local
    lap = 0
    net = Neural_Network('pong', network_path='{0}/Model_{1}'.format('pong', lap-1)) # if lap > 0
    agent = Agent("pong", "pong", net)
    Fetch_trajectories(agent, beta=0.6**lap)

    # to update data_set on VM (exec on VM)
    # sudo rm -r /home/jbremy/Imitation_Learning/pong/images
    # sudo rsync -au charlesdognin@83.202.87.74:/home/desktop/Imitation_Learning/pong/images/ /home/jbremy/Imitation_Learning/pong/images/
    # sudo rm -r /home/jbremy/Imitation_Learning/pong/actions
    # sudo scp -r charlesdognin@2a01:cb04:507:800:e51d:98c7:6eed:a687:/home/desktop/Imitation_Learning/pong/actions/ /home/jbremy/Imitation_Learning/pong/actions/

    # to exec in VM
    import parameters
    from network import Neural_Network
    import tensorflow as tf
    import time
    lap=0
    writer = tf.summary.FileWriter('{0}/Model_{1}/logs/train/'.format('pong', lap))
    net = Neural_Network('CarRacing')
    net.fit('/GPU:0', Data_path='pong', save_path="pong/Model_{}".format(lap), writer=writer, start_time=time.time(), lap=lap)
    writer.close()

    # update network in local
    # sudo scp -r jbremy@40.65.114.58:/home/jbremy/Imitation_Learning/pong/Model_!!!!lap/ /home/desktop/Imitation_Learning/pong/Model_!!!!lap/
