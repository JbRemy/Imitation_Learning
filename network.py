'''
Implementation of the network thats learns ono the expert policies data set
'''

import tensorflow as tf
import parameters
import numpy as np
import time
import os

from utils import variable_summaries

class Neural_Network(object):

    def __init__(self, game, network_path=None):
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
        self.learning_rate = self.parameters['learning_rate']
        self.n_epochs = self.parameters['n_epochs']
        self.batch_size = self.parameters['batch_size']
        self.n_hidden_layers = self.parameters['n_hidden_layers']
        self.n_hidden_layers_nodes = self.parameters['n_hidden_layers_nodes']
        self.optimizer = self.parameters['optimizer']

        self.n_input_features = int(self.input_H*self.input_W/6)*self.input_C*4

        if game == 'pong' or game == 'CarRacing':
            self.build_model = self._build_network_full_images
            self.placeholders = self._placeholders_full_images
            self.predict_function = self.predicit_full_images

        if isinstance(network_path, str):
            self.network_path = network_path


    def fit(self, device, Data_path, save_path, writer, start_time, lap):
        '''
        Fits the Network to the set currently in Data/$game
        :param device: (str) '/GPU:0' or '/CPU:0'
        :param Data_path: (str) path towerds the training set
        :param save_path: (str) path to save the trained model, learning curves are to be saved with the global writer
        :param writer: (tf.summary.FileWriter) the global file writer that saves all learning curves
        :param start_time: (time) current time
        '''

        self.network_path = save_path
        self.set_size = len(os.listdir('{}/images'.format(Data_path)))
        print('Sarting Lap : {} Training ...')
        print(' -- Initializing network ...')
        with tf.device(device):
            acc = 0
            tf.summary.scalar('accuracy', acc, family='Lap_{}'.format(lap))
            with tf.name_scope('Lap_{}'.format(lap)):
                with tf.name_scope('Inputs'):
                    X, y, keep_prob = self.placeholders(self.n_actions)

                with tf.name_scope('Layers'):
                    network = self.build_model(X, keep_prob, self.n_input_features, self.n_hidden_layers_nodes,
                                                                         self.n_actions, lap)

                with tf.name_scope('Loss'):
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=network,
                                                                                  name='Entropy'), name='Reduce_mean')
                    tf.summary.scalar('loss', loss, collections=['train'], family='Lap_{}'.format(lap))
                    merged_summary = tf.summary.merge_all('train')

                with tf.name_scope('Optimizer'):
                    global_step = tf.Variable(0, trainable=False)
                    learning_rate_decay = tf.train.exponential_decay(self.learning_rate, global_step,
                                                                     1000, 0.5, staircase=True, name='Learning_Rate')
                    optimizer = self.optimizer(learning_rate_decay)
                    minimizer = optimizer.minimize(loss)

                print(' -- Network initialized')
                print(' -- Starting training ...')
                if lap == 0:
                    writer.add_graph(tf.get_default_graph())


                saver = tf.train.Saver()
                init = tf.global_variables_initializer()
                X_temp = np.zeros((1, self.input_W, self.input_H, self.input_C*4))
                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                    sess.run(init)
                    for epoch in range(self.n_epochs):
                        for batch_number in range(int(self.set_size/self.batch_size)):
                            X_batch, y_batch =  self._make_batch_full_images(self.game, Data_path, self.batch_size, self.n_actions)
                            cost = 0
                            for ind in range(self.batch_size):
                                X_temp[0, :, :, :] = X_batch[ind, :, :, :]
                                _, res, c, summary = sess.run([minimizer, network, loss, merged_summary], feed_dict={X: X_temp, y: y_batch[ind, :, :], keep_prob: 0.7})
                                cost += c
                                if np.argmax(res) == np.argmax(y_batch[_, :]):
                                    acc += 1

                            if batch_number % 10 == 0:
                                writer.add_summary(summary, batch_number + epoch * int(self.set_size/self.batch_size))
                                writer.add_summary(summary, batch_number + epoch * int(self.set_size / self.batch_size))
                                print(' |-- Epoch {0} Batch {1} done ({2}) :'.format(epoch, batch_number, time.strftime("%H:%M:%S",
                                                                                        time.gmtime(time.time() - start_time))))
                                print(' |--- Avg cost = {}'.format(cost/self.batch_size))

                    saver.save(sess, '{}/model.ckpt'.format(save_path))
                    sess.close()

            #writer.add_summary(acc, lap)
            print(' -- Training over')
            print(' -- Model saved to : {}'.format(save_path))

    def predict(self, sess, X_feed, X_train, keep_prob, out):
        '''
        Predicts the output of the network for a set of images
        :param sess: a running session
        :param X_feed: (np array)
        :param X_train: (tensor placeholder)
        :param keep_prob: (tensor placeholder)
        :param out: (tensor placeholder)
        :return:
        '''

        res = sess.run(out, feed_dict={X_train: X_feed, keep_prob: 1})
        out = np.argmax(res)

        return out


    def _make_batch_full_images(self, Data_path, batch_size, n_actions):
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

        num_states = len(os.listdir('{}/images'.format(Data_path)))
        batch_list = np.random.choice(num_states-1, batch_size)
        X_batch = np.zeros([batch_size, self.input_W, self.input_H, self.input_C*4])
        y_batch = np.zeros([batch_size, 1, n_actions])
        for _, i in enumerate(batch_list):
            img = np.load('{0}/images/state_{1}.npy'.format(Data_path, i+1))
            for _2 in range(4):
                X_batch[_, :, :, _2*self.input_C:(_2+1)*self.input_C] = img[_2, :, :, :]/255

            y_batch[_, :, :] = np.transpose(np.load('{0}/actions/state_{1}.npy'.format(Data_path, i+1)))

        return X_batch, y_batch


    def _build_network_full_images(self, input, keep_prob, n_features, n_hidden_layer_nodes, output_size, lap):
        '''
        Builds a one layer neural network. For image only features.
        :param input: (tensor)
        :param n_features: (int)
        :param n_hidden_layer_nodes: (int) number of nodes in the hidden layer
        :param output_size: (int)
        :param training: (Boolean) if training dropout is activated
        :return: (tensor)
        '''

        pooled = tf.layers.max_pooling2d(input, pool_size=(3, 2), strides=(3, 2), name='Max_Pooling_layer',)
        flat_pooled = tf.contrib.layers.flatten(pooled)
        hidden_out = self._linear_layer(flat_pooled, n_features, n_hidden_layer_nodes, name='Hidden_Layer')
        with tf.name_scope('Dropout'):
            hidden_out = tf.nn.dropout(hidden_out, keep_prob=keep_prob, name='Dropout')

        out = self._linear_layer(hidden_out, n_hidden_layer_nodes, output_size, name='Output', lap=lap)

        return out


    def _linear_layer(self, input, dim_0, dim_1, name='', out_layer=False, lap=0):
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
            with tf.name_scope('Weights'):
                W =  tf.Variable(tf.truncated_normal([dim_0, dim_1], stddev=0.1), name='W')
                variable_summaries(W, ['train'], family='Lap_{}'.format(lap))
                b = tf.Variable(tf.zeros([1, dim_1]), name='Bias_hidden')
                variable_summaries(b, ['train'], family='Lap_{}'.format(lap))

            out_matmul = tf.matmul(input, W, name='Matmul')
            out = tf.add(out_matmul, b, name='Add')
            tf.summary.histogram('pre_activations', out, collections=['train'], family='Lap_{}'.format(lap))
            if out_layer == False:
                out = tf.nn.relu(out, name='Relu')
                tf.summary.histogram('post_activations', out, collections=['train'], family='Lap_{}'.format(lap))

        return out


    def _placeholders_full_images(self, n_actions):
        '''
        Creates placeholders. For image only features.
        :param n_features: (int)
        :param n_actions:  (int)
        :return: (tensor) (tensor) (tensor)
        '''

        X = tf.placeholder(dtype='float32', shape=[1, self.input_W, self.input_H, self.input_C*4], name='X_train')
        y = tf.placeholder(dtype='float32', shape=[1, n_actions], name='y_train')
        keep_prob = tf.placeholder(tf.float32, name='Keep_Prob')

        return X, y, keep_prob

if __name__=="__main__":
    net = Neural_Network('pong')
    writer = tf.summary.FileWriter('pong/Model/logs/train/')
    net.fit('/CPU:0', Data_path="pong", save_path="pong/Model", writer=writer, start_time=time.time(), lap=0)
    writer.close()