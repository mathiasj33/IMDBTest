from data_loader import DataLoader, DataType
import tensorflow as tf
import numpy as np

dl = DataLoader()


def weight_variable(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)


def get_accuracy(sess, x, y, y_):
    correct_prediction = tf.equal(tf.round(y), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_data = [np.array(t) for t in zip(*dl.load_labelled_vectors(DataType.TEST))]
    return sess.run(accuracy, feed_dict={x: test_data[0], y_: test_data[1]})


def train():
    x = tf.placeholder(tf.float32, [None, 500])

    # hidden layer
    W_fc1 = weight_variable([500, 30])
    b_fc1 = bias_variable([30])
    h_fc1 = tf.sigmoid(tf.matmul(x, W_fc1) + b_fc1)

    # output layer
    W_fc2 = weight_variable([30, 1])
    b_fc2 = bias_variable([1])
    y = tf.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

    y_ = tf.placeholder(tf.float32, [None, 1])
    loss = tf.reduce_mean(tf.reduce_sum(tf.pow(y_ - y, 2), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(10).minimize(loss)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    epochs = 30
    mini_batch_size = 10
    train_size = 1200
    for i in range(epochs):
        mini_batches = [np.array(t) for t in zip(*dl.mini_batches(mini_batch_size))]
        for mini_batch in mini_batches:
            sess.run(train_step, feed_dict={x: mini_batch[0], y_: mini_batch[1]})
        print("Epoch {} accuracy: {:.2f}%".format(i + 1, get_accuracy(sess, x, y, y_) * 100))

    print("Final accuracy: {:.2f}%".format(get_accuracy() * 100))


if __name__ == '__main__':
    train()
