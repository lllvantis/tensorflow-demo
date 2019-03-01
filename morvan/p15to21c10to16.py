import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer_%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]), name='weight')
            tf.summary.histogram(layer_name + '_weights', weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='bias')
            tf.summary.histogram(layer_name + '_biases', biases)
        with tf.name_scope('w_x_plus_b'):
            w_x_plus_b = tf.matmul(inputs, weights) + biases
            tf.summary.histogram(layer_name + '_w_x_plus_b', w_x_plus_b)
        if activation_function is None:
            outputs = w_x_plus_b
        else:
            outputs = activation_function(w_x_plus_b)
        tf.summary.histogram(layer_name + '_outputs', outputs)
        return outputs


# make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)


with tf.name_scope('losses'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]), name='loss')
    tf.summary.scalar('loss', loss)

with tf.name_scope('trains'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # learning_rate < 1

init = tf.global_variables_initializer()

sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
merged = tf.summary.merge_all()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i < 10 or i % 50 - 1 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)
        plt.pause(0.1)
plt.ioff()
plt.show()
