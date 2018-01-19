'''
Utilities
'''

import tensorflow as tf
from gym.utils import play

import os

import gym
import pygame
import sys
import time
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy.random import uniform
from skimage.measure import block_reduce
import time

from collections import deque
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, VIDEORESIZE


def variable_summaries(var, collections, family):
    '''
    Saves metrics about a variable
    :param var:
    :return:
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean, collections='Variables', family=family)

    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

    tf.summary.scalar('stddev', stddev, collections=collections, family=family)
    tf.summary.scalar('max', tf.reduce_max(var), collections=collections, family=family)
    tf.summary.scalar('min', tf.reduce_min(var), collections=collections, family=family)
    tf.summary.histogram('histogram', var, collections=collections)


def Fetch_trajectories(agent, lap, beta=1):
    '''
    Fetch trajectories from a mix of the expert and the learned policy
    :param agent: (agent.agent)
    :param beta: (float) the mixing rate
    '''

    if beta == 1:
        play_expert_agent_humans(agent.env, agent.policy, agent.n_actions, agent.data_path, beta,
                                 callback=save_state, keys_to_action=agent.keys_to_action, action_list=agent.list_action)

    else:
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver = tf.train.import_meta_graph('{}/model.ckpt.meta'.format(agent.Network.network_path))
            saver.restore(sess, tf.train.latest_checkpoint(agent.Network.network_path))
            graph = tf.get_default_graph()
            X_train = graph.get_tensor_by_name('Lap_{}/Inputs/X_train:0'.format(lap))
            keep_prob = graph.get_tensor_by_name('Lap_{}/Inputs/Keep_Prob:0'.format(lap))
            out = graph.get_tensor_by_name('Lap_{}/Layers/Output/Add:0'.format(lap))

            play_expert_agent_humans(agent.env, lambda x: agent.policy(x, sess, X_train,keep_prob, out),
                                     agent.n_actions, agent.data_path, beta=beta, transpose=True, fps=20, zoom=3,
                                     callback=save_state, keys_to_action=agent.keys_to_action, action_list=agent.list_action)


def save_state(previous_states, action, save_path):
    '''
    saves staes
    :param previous_states: (np array) current image and 3 previous one
    :param action: (np array) action chosen by the expert
    :param save_path: (str)
    '''

    state_number = len(os.listdir('{}/images'.format(save_path)))
    previous_states_save = np.zeros([4, int(previous_states.shape[1] / 2), int(previous_states.shape[2] / 2), 3])
    for _ in range(4):
        previous_states_save[_, : , :, :] = block_reduce(previous_states[_, : , :, :], (2, 2, 1), np.max)

    np.save('{}/images/state_{}'.format(save_path, state_number), previous_states)
    np.save('{}/actions/state_{}'.format(save_path, state_number), action)


def play_expert_agent_humans(env, agent_policy, n_actions, data_set_path, beta, transpose=True, fps=20, zoom=3, callback=None,
                             callback_2=None, keys_to_action=None, action_list = []):
    '''
    This function is an adaptation of the gym.utils.play function that allows the agent to play in place of the expert,
    and to save the states.
    :param env: (gym.Env)
    :param agent_policy: (network.predict)
    :param data_set_path: (str)
    :param proba: (float) proba to fetch expert
    :param transpose: (Boolean) If True the output of observation is transposed.
    :param fps: (int) frames per seconds
    :param zoom: (float) Make screen edge this many times bigger
    :param callback: here we use callback to save the trajectory
    :param keys_to_action: (dict)
            {
                # ...
                sorted(ord('w'), ord(' ')) -> 2
                # ...
            }
    '''

    obs_s = env.observation_space
    assert type(obs_s) == gym.spaces.box.Box
    assert len(obs_s.shape) == 2 or (len(obs_s.shape) == 3 and obs_s.shape[2] in [1, 3])

    if keys_to_action is None:
        if hasattr(env, 'get_keys_to_action'):
            keys_to_action = env.get_keys_to_action()

        elif hasattr(env.unwrapped, 'get_keys_to_action'):
            keys_to_action = env.unwrapped.get_keys_to_action()

        else:
            assert False, env.spec.id + " does not have explicit key to action mapping, " + \
                          "please specify one manually"
    relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

    if transpose:
        video_size = env.observation_space.shape[1], env.observation_space.shape[0]

    else:
        video_size = env.observation_space.shape[0], env.observation_space.shape[1]

    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    pressed_keys = []
    running = True
    env_done = True

    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()

    count = np.zeros((n_actions,1))

    while running:
        if env_done:
            env_done = False
            obs = env.reset()
            previous_obs = np.stack([obs for _ in range(4)])
            cum_rew = 0

        else:
            u = uniform()
            if u < beta:

                action = keys_to_action[tuple(sorted(pressed_keys))]
                obs, rew, env_done, info = env.step(action)

            else:
                action = action_list[agent_policy(previous_obs)]
                print(action)
                obs, rew, env_done, info = env.step(action)
                action = keys_to_action[tuple(sorted(pressed_keys))]

            action_out = np.zeros((n_actions, 1))
            action_out[np.array(action_list) == action, :] = 1

            previous_obs[3, :, :, :] = previous_obs[2, :, :, :]
            previous_obs[2, :, :, :] = previous_obs[1, :, :, :]
            previous_obs[1, :, :, :] = previous_obs[0, :, :, :]
            previous_obs[0, :, :, :] = obs

            cum_rew += rew

            if callback is not None:
                callback(previous_obs, action_out, data_set_path)
                count += action_out

            if callback_2 is not None:
                callback_2(obs_t=None, obs_tp1=None, action=None, rew=None, done=None, info=None, cum_rew=cum_rew)

        if obs is not None:
            if len(obs.shape) == 2:
                obs = obs[:, :, None]

            if obs.shape[2] == 1:
                obs = obs.repeat(3, axis=2)

            display_arr(screen, obs, transpose=transpose, video_size=video_size)

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:
                    pressed_keys.append(event.key)
                    #print(pressed_keys)
                elif event.key == 27:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in relevant_keys:
                    pressed_keys.remove(event.key)
                    #print(pressed_keys)
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                #print(video_size)

        pygame.display.flip()
        pygame.time.wait(10)
        clock.tick(fps)

    pygame.quit()
    #print(count)

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))

class PlayPlot(object):
    def __init__(self, callback, horizon_timesteps, plot_names):
        self.data_callback = callback
        self.horizon_timesteps = horizon_timesteps
        self.plot_names = plot_names

        num_plots = len(self.plot_names)
        self.fig, self.ax = plt.subplots(num_plots)
        if num_plots == 1:
            self.ax = [self.ax]
        for axis, name in zip(self.ax, plot_names):
            axis.set_title(name)
        self.t = 0
        self.cur_plot = [None for _ in range(num_plots)]
        self.data     = [deque(maxlen=horizon_timesteps) for _ in range(num_plots)]

    def callback(self, obs_t, obs_tp1, action, rew, done, info, cum_rew):
        points = self.data_callback(obs_t, obs_tp1, action, rew, done, info, cum_rew)
        for point, data_series in zip(points, self.data):
            data_series.append(point)
        self.t += 1

        xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t

        for i, plot in enumerate(self.cur_plot):
            if plot is not None:
                plot.remove()
            self.cur_plot[i] = self.ax[i].scatter(range(xmin, xmax), list(self.data[i]))
            self.ax[i].set_xlim(xmin, xmax)
        plt.pause(0.1)


if __name__ == "__main__":
    from utils import Fetch_trajectories
    from agent import Agent
    agent = Agent("CarRacing", "/Users/charlesdognin/Desktop/Imitation_Learning/pong", 0)
    keys_to_actions = {(): 0, (0,): 1, (1,): 2}
    Fetch_trajectories(agent, beta=1)