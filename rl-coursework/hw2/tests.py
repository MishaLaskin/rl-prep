from train_pg_f18 import build_mlp
import numpy as np
import tensorflow as tf


def mlp_test():

    input_size = 111
    output_size = 1
    n_layers = 2
    size = 64
    N = 1
    with tf.variable_scope("data"):
        x_ph = tf.placeholder(dtype=tf.float32, shape=(None, input_size))
        y_ph = tf.placeholder(dtype=tf.float32, shape=(None, output_size))

    x_data = np.random.random_sample((N, input_size))
    y_data = np.random.random_sample((N, output_size))

    model = build_mlp(x_ph, output_size, "data", n_layers,
                      size, activation=tf.tanh, output_activation=None)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_pred = sess.run(model, feed_dict={x_ph: x_data})
        print('build_mlp test')
        print(y_pred.shape, y_data.shape)


def discount_sum(rews, gamma, rewards_to_go=False):
    n = len(rews)
    discounted_rews = [gamma ** t * r for t, r in enumerate(rews)]
    if rewards_to_go:
        return [sum(discounted_rews[t:]) for t in range(n)]
    else:
        return [sum(discounted_rews) for t in range(n)]


mlp_test()
