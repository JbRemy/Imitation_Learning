import os
import numpy as np
from sklearn import datasets
from sklearn import model_selection
import matplotlib.pyplot as plt
import tensorflow as tf


def create_batch(batch_size, x_data, y_data):
    batch_mask = np.random.choice(len(x_data), batch_size, replace=False)
    x_batch = x_data[batch_mask]
    y_batch = y_data[batch_mask]
    return x_batch, y_batch


def to_one_hot(y):
    classes = np.unique(y)
    one_hot = np.array([[int(i == c) for c in classes] for i in y])
    return one_hot


class Data:
    def __init__(self, n, seed):
        [self.X, self.y] = datasets.make_moons(n_samples=n, noise=.1, random_state=seed)
        self.y_hot = to_one_hot(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test =\
            model_selection.train_test_split(self.X, self.y_hot, random_state=42, test_size=0.3)

saver = tf.train.Saver()

class NeuralNetwork:

    def __init__(self, size_hidden_layers, output_size, learning_rate, sess,
                 input_size=(224, 224, 3), batch_size=32, epochs=50, seed=None):

        if not (seed is None):
            self.rng = np.random.RandomState(seed)
        self.sess=sess
        self.batch_size = batch_size
        self.input_size = input_size
        self.epochs = epochs
        self.x = tf.placeholder(tf.float32, [None, input_size])
        self.y = tf.placeholder(tf.float32, [None, output_size])

        self.weights = {
            'hidden': tf.Variable(tf.random_normal([input_size, size_hidden_layers], seed=seed)),
            'output': tf.Variable(tf.random_normal([size_hidden_layers, output_size], seed=seed))
        }

        self.biases = {
            'hidden': tf.Variable(tf.random_normal([size_hidden_layers], seed=seed)),
            'output': tf.Variable(tf.random_normal([output_size], seed=seed))
        }

        self.hidden_layer = tf.add(tf.matmul(self.x, self.weights['hidden']), self.biases['hidden'])
        self.hidden_layer = tf.nn.relu(self.hidden_layer)

        self.output_layer = tf.matmul(self.hidden_layer, self.weights['output']) + self.biases['output']

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_layer, labels=self.y))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)

        self.init = tf.global_variables_initializer()


    def fit(self, X, y):

        with tf.Session() as self.sess:
            self.sess.run(self.init)

            for epoch in range(self.epochs):
                avg_cost = 0
                total_batch = int(X.shape[0] / self.batch_size)
                for i in range(total_batch):
                    batch_x, batch_y = create_batch(self.batch_size, X, y)
                    _, c = self.sess.run([self.optimizer, self.cost], feed_dict={x: batch_x, y: batch_y})

                    avg_cost += c / total_batch

                if epoch % 10 == 0:
                    print("Epoch: %i, cost = %f" % (epoch + 1, avg_cost))

            print("\nTraining complete!")

    def save_model(self, path="/tmp/model.ckpt"):
        self.save_path = saver.save(self.sess, path)
        print("Model saved in file: %s" % self.save_path)


    def predict(self, X):

        with tf.Session() as sess:
            saver.restore(sess, self.save_path)

            predict = tf.argmax(self.output_layer, 1)
            prediction = predict.eval({self.x: X})

        return prediction


if __name__ == "__main__":
    print("Testing Neural Network with two_moons")

    seed = 42
    rng = np.random.RandomState(seed)

    # generating data
    data = Data(1000, seed)

    plt.scatter(data.X[:, 0], data.X[:, 1], c=data.y)

    NN = NeuralNetwork(input_size=2, size_hidden_layers=32, output_size=2,  learning_rate=0.01, seed=seed)
    NN.train( X=data.X_train, y=data.y_train, num_epochs=100, batch_size=128, save_file="/tmp/model.ckpt")
    pred = NN.make_prediction(data.X_test)

    plt.scatter(data.X_test[:, 0], data.X_test[:, 1], c=pred)
