#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from ..optmizers import get_best_model
import pickle
import onnxmltools


class SklearnKerasClassifier(KerasClassifier):
        def __init__(self, build_fn, **kwargs):
            super(KerasClassifier, self).__init__(build_fn=build_fn)
            self.primal = kwargs['primal']
            self.params = kwargs['params']
            self.best = kwargs['best']

        def fit(self, x_train, y_train, **kwargs):
            import numpy as np
            kwargs.setdefault('verbose', 0)
            verbose = kwargs['verbose']
            kwargs.setdefault('params', self.params)
            kwargs.setdefault('space', False)

            y_train = np.array(y_train) # Compatibility with all formats?
            if len(y_train.shape) == 2 and y_train.shape[1] > 1:
                self.classes_ = np.arange(y_train.shape[1])
            elif (len(y_train.shape) == 2 and y_train.shape[1] == 1) or len(y_train.shape) == 1:
                self.classes_ = np.unique(y_train)
                y_train = np.searchsorted(self.classes_, y_train)
            else:
                raise ValueError(
                        'Invalid shape for y_train: ' + str(y_train.shape))
            # Check whether to compute for the best model or not
            if (self.best and kwargs['params'] != self.params and kwargs['space']):
                # Optimize
                hyperopt_space = kwargs['space']
                self.params.update(kwargs['params'])

                primal_model = self.primal
                primal_model.fit(x_train, y_train)
                y_pred = primal_model.predict(x_train)
                primal_data = {
                    'y_pred': y_pred,
                    'model_name': primal_model.__class__.__name__
                }

                ## Search for best model using Tune ##
                self.model = get_best_model(x_train, y_train,
                    primal_data=primal_data, params=self.params, space=hyperopt_space)
                self.model.fit(x_train, y_train, epochs=200,
                            batch_size=30, verbose=verbose) # Not necessary. Fitting twice by now.
                return self.model # Not necessary.
            else:
                # Dont Optmize
                primal_model = self.primal
                primal_model.fit(x_train, y_train)
                if primal_model.__class__.__name__ is not 'DecisionTreeClassifier':
                    self.model = self.build_fn.__call__(x_train=x_train)
                    y_pred = primal_model.predict(x_train)
                    self.model.fit(x_train, y_pred, epochs=500, batch_size=500, verbose=verbose)

                else:
                    feature_index, count = np.unique(primal_model.tree_.feature, return_counts=True)
                    cuts_per_feature = np.zeros(shape=x_train.shape[1], dtype=int)

                    cuts_per_feature = [count[j] for i,_ in enumerate(cuts_per_feature)
                                        for j, value in enumerate(feature_index) if i==value]

                    cuts_per_feature = [1 if value < 1 else np.ceil(x_train.shape[0]) if value > np.ceil(x_train.shape[0]) else count[i]
                                        for i, value in enumerate(count)]

                    # for i, value in enumerate(cuts_per_feature):
                    #     if value < 1:
                    #         cuts_per_feature[i] = 1
                    #     elif value > np.ceil(x_train.shape[0]):
                    #         cuts_per_feature[i] = np.ceil(x_train.shape[1]) ## Convert to list comprehension
                    #     else:
                    #         cuts_per_feature[i] = count[i]
                    # for i,_ in enumerate(cuts_per_feature):
                    #     for j,value in enumerate(feature_index):
                    #         if i==value:
                    #             cuts_per_feature[i] = count[j]

                    units = y_train.shape[1]
                    self.model = self.build_fn.__call__(x_train=x_train, cuts_per_feature=cuts_per_feature,
                                                        units=units)
                    self.model.fit(x_train, y_train, epochs=500, batch_size=500, verbose=verbose)

                return self.model # Not necessary.

        def save(self, filename = None):
            if filename == None:
                raise ValueError('Name Error: to save the model you need to specify the filename')

            pickle.dump(self.model, open(filename + '.pkl', 'wb'))

            self.model.save(filename + '.h5')

            onnx_model = onnxmltools.convert_keras(self.model)
            onnxmltools.utils.save_model(onnx_model, filename + '.onnx')


class SklearnKerasRegressor(KerasRegressor):
        def __init__(self, build_fn, **kwargs):
            super(KerasRegressor, self).__init__(build_fn=build_fn)
            self.primal = kwargs['primal']
            self.params = kwargs['params']
            self.best = kwargs['best']

        def fit(self, x_train, y_train, **kwargs):
            # Check whether to compute for the best model or not
            kwargs.setdefault('verbose', 0)
            verbose = kwargs['verbose']
            if (self.best and self.params != None):
                # Optimize
                primal_model = self.primal
                primal_model.fit(x_train, y_train)
                y_pred = primal_model.predict(x_train)
                primal_data = {
                    'y_pred': y_pred,
                    'model_name': primal_model.__class__.__name__
                }

                self.model, final_epoch, final_batch_size = get_best_model(
                    x_train, y_train, primal_data=primal_data,
                    params=self.params, build_fn=self.build_fn)
                # Epochs and batch_size passed in Talos as well
                self.model.fit(x_train, y_train, epochs=final_epoch,
                    batch_size=final_batch_size, verbose=verbose)
                return self.model
            else:
                # Dont Optmize
                self.model = self.build_fn.__call__(x_train=x_train)
                self.model.fit(x_train, y_train, epochs=500, batch_size=500)
                return self.model

        def score(self, x, y, **kwargs):
            score = super(SklearnKerasRegressor, self).score(x, y, **kwargs)
            # keras_regressor treats all score values as loss and adds a '-ve' before passing
            return -score

        def save(self, filename=None):
            if filename == None:
                raise ValueError(
                    'Name Error: to save the model you need to specify the filename')

            pickle.dump(self.model, open(filename + '.pkl', 'wb'))

            self.model.save(filename + '.h5')

            onnx_model = onnxmltools.convert_keras(self.model)
            onnxmltools.utils.save_model(onnx_model, filename + '.onnx')


class SklearnTensorflowClassifier():
        def __init__(self, build_fn, **kwargs):
            self.build_fn = build_fn
            self.primal = kwargs['primal']
            self.params = kwargs['params']
            self.best = kwargs['best']

        def score(self, x, y):
           import numpy as np
           # TODO
           # Fix the access flow of y_pred and x_ph
           return np.mean(np.argmax(self.y_pred.eval(feed_dict={self.x_ph: x}), axis=1) == np.argmax(y, axis=1))

        def fit(self, x_train, y_train, **kwargs):
            import numpy as np
            import tensorflow as tf
            kwargs.setdefault('batch_size', 500)
            kwargs.setdefault('epochs', 500)
            kwargs.setdefault('verbose', 0)
            verbose = kwargs['verbose']
            kwargs.setdefault('params', self.params)
            kwargs.setdefault('space', False)

            y_train = np.array(y_train) # Compatibility with all formats?
            if len(y_train.shape) == 2 and y_train.shape[1] > 1:
                self.classes_ = np.arange(y_train.shape[1])
            elif (len(y_train.shape) == 2 and y_train.shape[1] == 1) or len(y_train.shape) == 1:
                self.classes_ = np.unique(y_train)
                y_train = np.searchsorted(self.classes_, y_train)
            else:
                raise ValueError(
                        'Invalid shape for y_train: ' + str(y_train.shape))
            # Check whether to compute for the best model or not
            # if (self.best and kwargs['params'] != self.params and kwargs['space']):
            #     # Optimize
            #     hyperopt_space = kwargs['space']
            #     self.params.update(kwargs['params'])

            #     primal_model = self.primal
            #     primal_model.fit(x_train, y_train)
            #     y_pred = primal_model.predict(x_train)
            #     primal_data = {
            #         'y_pred': y_pred,
            #         'model_name': primal_model.__class__.__name__
            #     }

            #     ## Search for best model using Tune ##
            #     self.model = get_best_model(x_train, y_train,
            #         primal_data=primal_data, params=self.params, space=hyperopt_space)
            #     self.model.fit(x_train, y_train, epochs=200,
            #                 batch_size=30, verbose=verbose) # Not necessary. Fitting twice by now.
            #     return self.model # Not necessary.
            # else:
                # Dont Optmize

            self.init_ops, optimizer, loss, self.y_pred, self.x_ph = self.build_fn.__call__(x_train=x_train)

            with tf.Session() as sess:
                # initialise the variables
                sess.run(self.init_ops)
                total_batch = int(len(y_train) / kwargs['batch_size'])
                for epoch in range(kwargs['epochs']):
                        avg_cost = 0
                        for i in range(total_batch):
                            # batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                            _, c = sess.run([optimizer, loss],
                                            feed_dict={x_ph: x_train, y_ph: y_train})
                            avg_cost += c / total_batch
                        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
                # print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

            # self.model.fit(x_train, y_train, epochs=500, batch_size=500, verbose=verbose)
            # return self.model # Not necessary.

        # TODO
        # Batch split while training
        # Placeholders for x and y

        def save(self, filename = None):
            if filename == None:
                raise ValueError('Name Error: to save the model you need to specify the filename')

            import tensorflow as tf
            from onnx_tf.frontend import tensorflow_graph_to_onnx_model

            with tf.gfile.GFile("frozen_graph.pb", "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                onnx_model = tensorflow_graph_to_onnx_model(graph_def,
                                                "fc2/add",
                                                opset=6)

                file = open("tf_test.onnx", "wb")
                file.write(onnx_model.SerializeToString())
                file.close()

            # onnx_model = onnxmltools.convert_keras(self.model)
            # onnxmltools.utils.save_model(onnx_model, filename + '.onnx')


wrappers = {
    'LogisticRegression': SklearnKerasClassifier,
    'LinearRegression': SklearnKerasRegressor,
    'LinearDiscriminantAnalysis': SklearnKerasClassifier,
    'DecisionTreeClassifier': SklearnKerasClassifier
}
