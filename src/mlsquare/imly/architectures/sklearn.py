#!/usr/bin/env python
# -*- coding: utf-8 -*-


# from tensorflow import set_random_seed
# from numpy.random import seed
# seed(3)
# set_random_seed(3)


def generic_linear_model(**kwargs):
    try:
        from keras.models import Sequential
        from keras.layers.core import Dense

        model_params = kwargs['model_params']
        model = Sequential()
        model.add(Dense(model_params['units'],
                        input_dim=kwargs['x_train'].shape[1],
                        activation=model_params['activation']))
        model.compile(optimizer=model_params['optimizer'],
                      loss=model_params['losses'],
                      metrics=['accuracy'])

        return model
    except ImportError:
        print ("keras is required to transpile the model")
        return False


def linear_discriminant_analysis(**kwargs):
    try:
        from keras.models import Sequential
        from keras.layers.core import Dense
        from keras.regularizers import l2
        from ..commons.losses import mse_in_theano

        model = Sequential()
        model.add(Dense(
            kwargs['params']['units'],
            input_dim=kwargs['x_train'].shape[1],
            activation=kwargs['params']['activation'][0],
            kernel_regularizer=l2(1e-5)
        ))
        model.compile(
            optimizer=kwargs['params']['optimizer'],
            loss=mse_in_theano,
            metrics=['accuracy']
        )

        return model
    except ImportError:
        print("keras is required to transpile the model")
        return False

def cart(**kwargs):

    import numpy as np
    import tensorflow as tf
    import ../../../datasets/iris.csv as iris
    # x = iris.feature[:, 2:4]  # use "Petal length" and "Petal width" only
    # y = iris.label
    # d = x.shape[1]
    from functools import reduce


    def tf_kron_prod(a, b):
        res = tf.einsum('ij,ik->ijk', a, b)
        res = tf.reshape(res, [-1, tf.reduce_prod(res.shape[1:])])
        return res


    def tf_bin(x, cut_points, temperature=0.1):
        # x is a N-by-1 matrix (column vector)
        # cut_points is a D-dim vector (D is the number of cut-points)
        # this function produces a N-by-(D+1) matrix, each row has only one element being one and the rest are all zeros
        D = cut_points.get_shape().as_list()[0]
        W = tf.reshape(tf.linspace(1.0, D + 1.0, D + 1), [1, -1])
        cut_points = tf.contrib.framework.sort(cut_points)  # make sure cut_points is monotonically increasing
        b = tf.cumsum(tf.concat([tf.constant(0.0, shape=[1]), -cut_points], 0))
        h = tf.matmul(x, W) + b
        res = tf.nn.softmax(h / temperature)
        return res


    def nn_decision_tree(x, cut_points_list, leaf_score, temperature=0.1):
        # cut_points_list contains the cut_points for each dimension of feature
        leaf = reduce(tf_kron_prod,
                    map(lambda z: tf_bin(x[:, z[0]:z[0] + 1], z[1], temperature), enumerate(cut_points_list)))
        return tf.matmul(leaf, leaf_score)

    # Pass cutpoints per feature as list - each element would be the count of cutpoints per feature
    num_cutpoints = kwargs['num_cutpoints'] 
    num_leaf = np.prod(np.array(num_cut) + 1)
    num_class = len(np.unique(kwargs['y_train']))

    sess = tf.InteractiveSession()

    x_ph = tf.placeholder(tf.float32, [None, kwargs['x_train'].shape[1]])
    y_ph = tf.placeholder(tf.float32, [None, num_class])

    cut_points_list = [tf.Variable(tf.random_uniform([i])) for i in num_cutpoints]
    leaf_score = tf.Variable(tf.random_uniform([num_leaf, num_class])) #Clarify how tf.random_variable works and use case in this context

    y_pred = nn_decision_tree(x_ph, cut_points_list, leaf_score, temperature=0.1)
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=y_pred, onehot_labels=y_ph))

    opt = tf.train.AdamOptimizer(0.1)
    train_step = opt.minimize(loss)

    init_ops = tf.global_variables_initializer()

    return init_ops

    # sess.run(tf.global_variables_initializer())

    # for i in range(1000):
    #     _, loss_e = sess.run([train_step, loss], feed_dict={x_ph: kwargs['x_train'], y_ph: kwargs['y_train']})
    #     if i % 200 == 0:
    #         print(loss_e)
    # print('error rate %.2f' % (1 - np.mean(np.argmax(y_pred.eval(feed_dict={x_ph: x}), axis=1) == np.argmax(y, axis=1))))


dispatcher = {
    'glm': generic_linear_model,
    'lda': linear_discriminant_analysis,
    'cart': cart
}
