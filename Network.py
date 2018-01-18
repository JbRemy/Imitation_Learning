'''
Implementation of the network thats learns ono the expert policies data set
'''

import tensorflow as tf
import parameters
import numpy as np
from numpy.random import choice
import time

from Utils import variable_summaries

class Neural_Network(object):

    def __init__(self, game, training_lap):
        '''
        Initialises Neural Network parameters, based on the game parameters
        :param game: (str) the game to be played
        :param training_lap: (int) the iteration in SMILE or DAGGER
        '''
        self.game = game
        self.parameters = getattr(parameters, game)

        self.input_W = self.parameters['input_W']
        self.input_H = self.parameters['input_H']
        self.input_C = self.parameters['input_C']
        self.n_actions = self.parameters['n_actions']
        self.past_memory = self.parameters['past_memory']
        self.learning_rate = self.parameters['learning_rate']
        self.n_epochs = self.parameters['n_epochs']
        self.batch_size = self.parameters['batch_size']
        self.n_hidden_layers = self.parameters['n_hidden_layers']
        self.n_hidden_layers_nodes = self.parameters['n_hidden_layers_nodes']
        self.optimizer = self.parameters['optimizer']

        self.n_input_features = self.input_H*self.input_W*self.input_C*self.past_memory

        if game == 'pong':
            self.build_model = self._build_network_full_images
            self.placeholders = self._placeholders_full_images
            self.predict_function = self.predicit_full_images

        with open('Data/{}/states.txt'.format(game), 'r') as file:
            lines = file.readlines()
            self.set_size = len(lines)

        self.training_lap = training_lap

    def fit(self, device, Data_path, save_path, writer, start_time):
        '''
        Fits the Network to the set currently in Data/$game
        :param device: (str) '/GPU:0' or '/CPU:0'
        :param Data_path: (str) path towerds the training set
        :param save_path: (str) path to save the trained model, learning curves are to be saved with the global writer
        :param writer: (tf.summary.FileWriter) the global file writer that saves all learning curves
        :param start_time: (time) current time
        '''

        writer.add
        self.network_path = save_path

        print('Sarting Lap : {} Training ...')
        print(' -- Initializing network ...')
        with tf.device(device):
            with tf.name_scope('Inputs'):
                X, y, training = self.placeholders(self.n_input_features, self.n_actions)

            with tf.name_scope('Layers'):
                network = self.build_model(X, self.n_input_features, self.n_hidden_layers_nodes,
                                                                     self.n_actions, training)

            with tf.name_scope('Loss'):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=network,
                                                                              name='Entropy'), name='Reduce_mean')
                tf.summary.scalar('loss', loss, collections=['train_{}'.format(self.training_lap)])
                merged_summary = tf.summary.merge_all('train_{}'.format(self.training_lap))

            with tf.name_scope('Optimizer'):
                optimizer = self.optimizer(self.learning_rate)
                minimizer = optimizer.minimize(loss)

            print(' -- Network initialized')
            print(' -- Starting training ...')
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                sess.run(init, {training: True})
                for epoch in range(self.n_epochs):
                    for batch_number in range(int(self.set_size/self.batch_size)):
                        X_batch, y_batch =  self._make_batch_full_images(self.game, Data_path, self.batch_size,
                                                                         self.n_input_features, self.n_actions, self.past_memory)
                        cost = 0
                        for ind in range(self.batch_size):
                            _, c, summary = sess.run([minimizer, loss, merged_summary], feed_dict={X: X_batch[ind, :], y: y_batch[ind, :]})
                            cost += c

                        if batch_number % 100 == 0:
                            writer.add_summary(summary, batch_number + epoch * int(self.set_size/self.batch_size))
                            writer.add_summary(summary, batch_number + epoch * int(self.set_size / self.batch_size))
                            print(' |-- Epoch {0} Batch {1} done ({2}) :'.format(epoch, batch_number, time.strftime("%H:%M:%S",
                                                                                    time.gmtime(time.time() - start_time))))
                            print(' |--- Avg cost = {}'.format(cost/self.batch_size))

                saver.save(sess, save_path)
                sess.close()

            print(' -- Training over')
            print(' -- Model saved to : {}'.format(save_path))

    def predict(self, X):
        '''
        predicts the ourput of the network.
        :param X:
        :return:
        '''

        return self.predict_function(X, self.n_input_features)


    def predicit_full_images(self, X, n_features):
        '''
        Predicts the output of the network for the data stored in X_path. . For image only features.
        :param X: (np array)
        :param n_features: (int) Number of input features
        :return: (np array)
        '''

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver = tf.train.import_meta_graph('{}/model.meta'.format(self.network_path))
            saver.restore(sess, tf.train.latest_checkpoint(self.network_path))
            graph = tf.get_default_graph()
            X_train = graph.get_tensor_by_name('inputs/X_train:0')
            out = graph.get_tensor_by_name('Layers/Out')
            X_flat = X.flatten().reshape(n_features, 1)
            action = sess.run(out, feed_dict={X_train: X_flat})
            sess.close()

        return action

    def _make_batch_full_images(self, game, Data_path, batch_size, n_features, n_actions, past_memory):
        '''
        Makes a batch from the currently saved data set. For image only features.
        :param game: (str) the game to be played
        :param Data_path: (str) the path towards the training set
        :param batch_size: (int) the size of the batch to make
        :param n_features: (int) number of input features
        :param n_actions: (int) number of output features
        :param past_memory: (int) number of images to take before the state
        :return: (np array) (np array)
        '''

        with open('{}/{}/states.txt'.format(Data_path, game), 'r') as file:
            lines = choice(file.readlines(), batch_size)

        X_batch = np.zeros([batch_size, n_features, 1])
        y_batch = np.zeros([batch_size, n_actions, 1])
        for _ in range(batch_size):
            img = np.load('Data/{0}/images/{1}.npy'.format(game, lines[_]))
            X_batch[_, :, 1] = img.flatten()
            y_batch[_, :, 1] = np.load('Data/{0}/states/{1}.npy'.format(game, lines[_]))
            
        return X_batch, y_batch


    def _build_network_full_images(self, input, n_features, n_hidden_layer_nodes, output_size, training):
        '''
        Builds a one layer neural network. For image only features.
        :param input: (tensor)
        :param n_features: (int)
        :param n_hidden_layer_nodes: (int) number of nodes in the hidden layer
        :param output_size: (int)
        :param training: (Boolean) if training dropout is activated
        :return: (tensor)
        '''

        hidden_out = self._linear_layer(input, n_hidden_layer_nodes, n_features, name='Hidden_Layer')
        with tf.name_scope('Dropout'):
            if training:
                hidden_out = tf.nn.dropout(hidden_out, keep_prob=0.9, name='Dropout')

        out = self._linear_layer(hidden_out, output_size, n_hidden_layer_nodes, name='Output')

        return out


    def _linear_layer(self, input, dim_0, dim_1, name='', out_layer=False):
        '''
        Builds a linear layer
        :param input: (tensor)
        :param dim_0: (int)
        :param dim_1: (int)
        :param name: (str) name of the layer
        :param out_layer: (Boolean) if True, no activation
        :return: (tensor)
        '''

        with tf.name_scope(name):
            with tf.name_scope('Weights', ['train_{}'.format(self.training_lap)]):
                W =  tf.Variable(tf.truncated_normal([dim_0, dim_1], stddev=0.1), name='W')
                variable_summaries(W, ['train_{}'.format(self.training_lap)])
                b = tf.Variable(tf.zeros([dim_1, 1]), name='Bias_hidden')
                variable_summaries(b, ['train_{}'.format(self.training_lap)])

            out_matmul = tf.matmul(W, input, name='Matmul')
            out = tf.add(out_matmul, b, name='Add')
            tf.summary.histogram('pre_activations', out, collections=['train_{}'.format(self.training_lap)])
            if out_layer == False:
                out = tf.nn.relu(out, name='Relu')
                tf.summary.histogram('post_activations', out, collections=['train_{}'.format(self.training_lap)])

        return out


    def _placeholders_full_images(self, n_features, n_actions):
        '''
        Creates placeholders. For image only features.
        :param n_features: (int)
        :param n_actions:  (int)
        :return: (tensor) (tensor) (tensor)
        '''

        X = tf.placeholder(dtype='float32', shape=[n_features, 1], name='X_train')
        y = tf.placeholder(dtype='float32', shape=[n_actions, 1], name='y_train')
        training = tf.placeholder(dtype=tf.bool, shape=())

        return X, y, training