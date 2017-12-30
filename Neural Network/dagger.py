import numpy as np
import gym
from Class_NN import NeuralNetwork

class DAGGER(object):
    def __init__(self, parameters):
        self.env = gym.make(parameters["env_name"])
        self.deep_q_net = NeuralNetwork(batch_size=parameters['batch_size'],
                              optimizer=parameters["optimizer"],
                              input_shape=parameters["input_shape"])
        self.agents = [Agent(deepnet=self.deep_q_net,
                             policy_name="epsilon-greedy",
                             agent_parameters=parameters["agent_parameters"])
                       for _ in range(parameters["nb_agents"])]
        self.input_shape = parameters["input_shape"]
        self.gamma = parameters["gamma"]

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

    def run_agents(self, agents, nb_episodes):
        simulations = []
        for agent in agents:
            simulations.append(self.run_episodes(agent, nb_episodes))
        return simulations

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