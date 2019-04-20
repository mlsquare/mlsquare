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
                self.model = self.build_fn.__call__(x_train=x_train)
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

        def score():
            pass

        def predict():
            pass

        def predict_proba():
            pass

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
            
            self.init_ops = self.build_fn.__call__(x_train=x_train)

            with tf.Session() as sess:
                # initialise the variables
                sess.run(self.init_ops)
                total_batch = int(len(y_train) / kwargs['batch_size'])
                for epoch in range(kwargs['epochs']):
                        avg_cost = 0
                        for i in range(total_batch):
                            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                            _, c = sess.run([optimiser, cross_entropy], 
                                        feed_dict={x: batch_x, y: batch_y})
                            avg_cost += c / total_batch
                        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
                print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

            # self.model.fit(x_train, y_train, epochs=500, batch_size=500, verbose=verbose)
            # return self.model # Not necessary.

        def save(self, filename = None):
            if filename == None:
                raise ValueError('Name Error: to save the model you need to specify the filename')

            pickle.dump(self.model, open(filename + '.pkl', 'wb'))

            self.model.save(filename + '.h5')
            
            onnx_model = onnxmltools.convert_keras(self.model)
            onnxmltools.utils.save_model(onnx_model, filename + '.onnx')


wrappers = {
    'LogisticRegression': SklearnKerasClassifier,
    'LinearRegression': SklearnKerasRegressor,
    'LinearDiscriminantAnalysis': SklearnKerasClassifier,
    'DecisionTreeClassifier': SklearnKerasClassifier
}
