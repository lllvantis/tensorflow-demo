import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
x_m = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
x_m_train, x_m_test, y_train, y_test = train_test_split(x_m, y, test_size=.3)


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    # add one more layer and return the output of this layer
    weights_m = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]))
    w_x_plus_b = tf.matmul(inputs, weights_m) + biases
    w_x_plus_b = tf.nn.dropout(w_x_plus_b, keep_prob)
    if activation_function is None:
        output = w_x_plus_b
    else:
        output = activation_function(w_x_plus_b)
    tf.summary.histogram(layer_name + '_outputs', output)
    return output


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])  # 8 * 8
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

# the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()

# summary writer goes here
train_writer = tf.summary.FileWriter('logs/train', sess.graph)
test_writer = tf.summary.FileWriter('logs/test', sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(500):
    sess.run(train_step, feed_dict={xs: x_m_train, ys: y_train, keep_prob: 0.5})
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: x_m_train, ys: y_train, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_result = sess.run(merged, feed_dict={xs: x_m_test, ys: y_test, keep_prob: 1})
        test_writer.add_summary(test_result, i)
        pass
