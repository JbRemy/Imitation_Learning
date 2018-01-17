import numpy as np


class Agent(object):

    def __init__(self, deepnet, policy_name, agent_parameters):
        self.policies = {"epsilon-greedy": self.epsilon_greedy}

        self.nn = deepnet
        self.policy = self.policies[policy_name]

        self.epsilon = agent_parameters["epsilon"]

    def epsilon_greedy(self, state, possible_actions):
        num_actions = possible_actions.n
        if np.random.uniform() < self.epsilon:
            action = possible_actions.sample()
        else:
            values = [self.nn.predict(state=state, action=i) for i in range(num_actions)]
            action = np.argmax(values)
        return action

    def act(self, state, possible_actions):
        return self.policy(state=state,
                           possible_actions=possible_actions)

