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


if __name__ == "__main__":
    seed = 42
    rng = np.random.RandomState(seed)

    # generating data
    data = Data(1000, seed)

    plt.scatter(data.X[:, 0], data.X[:, 1], c=data.y)
    plt.scatter(data.X_test[:, 0], data.X_test[:, 1])

    # Parameters of the NN
    input_size = 2
    num_hidden_layers = 500
    output_size = 2

    epochs = 200
    batch_size = 128
    learning_rate = 0.01

    # Building the NN
    x = tf.placeholder(tf.float32, [None, input_size])
    y = tf.placeholder(tf.float32, [None, output_size])

    weights = {
        'hidden': tf.Variable(tf.random_normal([input_size, num_hidden_layers], seed=seed)),
        'output': tf.Variable(tf.random_normal([num_hidden_layers, output_size], seed=seed))
    }

    biases = {
        'hidden': tf.Variable(tf.random_normal([num_hidden_layers], seed=seed)),
        'output': tf.Variable(tf.random_normal([output_size], seed=seed))
    }

    hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
    hidden_layer = tf.nn.relu(hidden_layer)

    output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # Session
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            avg_cost = 0
            total_batch = int(data.X_train.shape[0] / batch_size)
            for i in range(total_batch):
                batch_x, batch_y = create_batch(batch_size, data.X_train, data.y_train)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

                avg_cost += c/(batch_size*total_batch)

            if epoch%10 == 0:
                print("Epoch: %i, cost = %f" % (epoch+1, avg_cost))

        print("\nTraining complete!")

        predict = tf.argmax(output_layer, 1)
        prediction = predict.eval({x: data.X_test})


    plt.scatter(data.X_test[:, 0], data.X_test[:, 1], c=prediction)