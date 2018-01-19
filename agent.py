import numpy as np
import tensorflow as tf
import gym
import network

from utils import play_expert_agent_humans, PlayPlot
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
        self.n_actions = self.parameters['n_actions']
        self.list_action = self.parameters['list_action']
        self.keys_to_action = self.parameters['keys_to_action']

    def play(self, plot_rew=False):
        '''
        plays the game
        :param plot_rew: (Boolean)
        '''

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver = tf.train.import_meta_graph('{}/model.meta'.format(self.Network.network_path))
            saver.restore(sess, tf.train.latest_checkpoint(self.Network.network_path))
            graph = tf.get_default_graph()
            X_train = graph.get_tensor_by_name('inputs/X_train:0')
            keep_prob = graph.get_tensor_by_name('inputs/Keep_Prob')
            out = graph.get_tensor_by_name('Layers/Out')

            if plot_rew:
                def callback(obs_t, rew, cum_rew, done, info):
                    return [cum_rew, ]

                env_plotter = PlayPlot(callback, 30 * 5, ["reward"])
                play_expert_agent_humans(self.env, lambda x: self.policy(x, sess, X_train, keep_prob, out),
                                         self.n_actions, beta=0, transpose=True, fps=20, zoom=3,
                                         callback=None, callback_2=env_plotter.callback,
                                         keys_to_action=self.keys_to_action,
                                         action_list=self.list_action)

            else:
                play_expert_agent_humans(self.env, lambda x: self.policy(x, sess, X_train, keep_prob, out),
                                         self.n_actions, beta=0, transpose=True, fps=20, zoom=3,
                                         callback=None, callback_2=None,
                                         keys_to_action=self.keys_to_action,
                                         action_list=self.list_action)


    def policy(self, previous_obs, sess, X_train, keep_prob, out):

        X_feed = np.zeros([1, self.parameters['input_W'], self.parameters['input_H'], self.parameters['input_C'] * 4])
        for _2 in range(4):
            X_feed[_, :, :, _2 * self.parameters['input_C']:(_2 + 1) * self.parameters['input_C']] = previous_obs[_2, :, :, :] / 255

            return self.Network.predict(sess, X_feed, X_train, keep_prob, out)
