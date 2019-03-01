# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    w_x_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = w_x_plus_b
    else:
        outputs = activation_function(w_x_plus_b)
    return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w_m):
    # strides: [batch_size, x_movement, y_movement, channel_count]
    # padding: SAME, VALID
    # must have strides[0] = strides[3], normally 1
    return tf.nn.conv2d(x, w_m, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # ksize: kernel size, size of pooling window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28 * 28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
print(x_image.shape)  # [n_sample, 28, 28, 1]

# conv 1 layer
w_m_conv1 = weight_variable([5, 5, 32, 32])  # patch = 5 * 5, in_size = 1, out_size = 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_m_conv1) + b_conv1)  # output size = 28 * 28 * 32
h_pool1 = max_pool_2x2(h_conv1)  # output size = 14 * 14 * 32

# conv 2 layer
w_m_conv2 = weight_variable([5, 5, 1, 64])  # patch = 5 * 5, in_size = 32, out_size = 64
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(x_image, w_m_conv2) + b_conv2)  # output size = 14 * 14 * 64
h_pool2 = max_pool_2x2(h_conv2)  # output size = 7 * 7 * 64

# func1 layer
w_m_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  # [n_samples, 7, 7, 64] -> [n_samples, 7 * 7 * 64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat))

# add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    train_images, train_labels = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: train_images, ys: train_labels})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
