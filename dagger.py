import numpy as np
import gym

import tensorflow as tf
import parameters
from Network import Neural_Network
from Utils import Fetch_trajectories
from agent import agent

class DAGGER(object):

    def __init__(self, game):

        self.game = game
        self.parameters = getattr(parameters, game)

        self.env = gym.make(self.parameters["env_name"])
        self.Network = Neural_Network(game)
        self.agent = agent(self.env, self.data_path, self.Network)
        self.path = self.parameters['path']

    def train(self, n_iterations):

        tf.summary.FileWriter('{}/logs/train/'.format(self.path))
        Fetch_trajectories(beta=1)



    def run_episodes(self, agent, nb_episodes):
        for i in range(nb_episodes):
            episodes_rewards = []
            episodes_obs = np.expand_dims(np.empty(self.input_shape), axis=0)
            episodes_actions = []

            reward = 0
            done = False
            ob = self.env.reset()

            while not done:
                action = agent.act(state=ob,
                                   possible_actions=self.env.action_space)
                ob, reward, done, _ = self.env.step(action)
                reward += reward

                episodes_rewards.append(reward)
                episodes_obs = np.concatenate((episodes_obs, np.expand_dims(ob, axis=0)), axis=0)
                episodes_actions.append(action)

        return {"rewards": np.array([sum(self.gamma**i * episodes_rewards[i::]) for i in range(len(episodes_rewards))]),
                "obs": episodes_obs,
                "actions": np.array(episodes_actions)}


    def run_simulator(self, nb_iterations, nb_episodes):
        for i in range(nb_iterations):
            print("Iteration " + str(i) + "/" + str(nb_iterations))
            simulations = self.reshape_simulations(self.run_agents(self.agents,
                                                                   nb_episodes=nb_episodes))
            self.deep_q_net.learn(x_train=simulations["obs"],
                                  y_train=simulations["rewards"])


    @staticmethod
    def reshape_simulations(simulations):
        rewards = np.concatenate(tuple([simulations["rewards"]]), axis=0)
        obs = np.concatenate(tuple([simulations["obs"]]), axis=0)
        return {"rewards": rewards,
                "obs": obs}