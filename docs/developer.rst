===============
Developer Guide
===============

Setup
=====

--------------------
Forking a repository
--------------------

To ensure a risk-free environment to work with, you will have to fork the mlsquare repository. Once you have
forked the repository, you can call `git fetch upstream` and `git pull 'branch-name'` before you make any local.
This will ensure that your local repository is up-to-date with the remote repository.

--------------------------------------------
Installing mlsquare after cloning repository
--------------------------------------------

After forking and updating your local repository, you might want to do the following to install the local repository
version. This will help you in testing changes you make to the repository.

.. code-block:: bash

   cd path-to-local-repo/mlsquare
   python setup.py develop

You can run the test cases using the following commands

.. code-block:: bash

   cd path-to-local-repo/mlsquare
   python setup.py test


Adding an algorithm
===================

This is for users interested in adding or contributing custom algorithms. Follow the below mentioned steps
to contribute to the existing collection of algorithms available in mlsquare.

-------------
Where to add?
-------------

Navigate to `mlsquare.architectures` folder. Choose your primal module(say `sklearn.py <https://github.com/mlsquare/mlsquare/blob/master/src/mlsquare/architectures/sklearn.py>`).
The `architectures` folder consists of all existing algorithm mappings. Each .py file in this folder represents a primal module.


--------------
Implementation
--------------

1. Each algorithm is expected to be declared as a class. An algorithm should be registered in the `registry`(check FAQs for details)
   using the @registry.register decorator.

2. Use the base class available in `base.py <https://github.com/mlsquare/mlsquare/blob/master/src/mlsquare/base.py#L43>` as the parent class for your algorithm. Feel free to use an already existing base class(ex - `glm <https://github.com/mlsquare/mlsquare/blob/master/src/mlsquare/architectures/sklearn.py#L16>`)
   if it matches your algorithm's needs.

3. The following methods and attributes are expected to implemented while creating a new model,
    `create_model()` - Your models architecture lies in this method. Calling this method would return a compiled
    dnn model(ex - keras or pytorch model).
    `set_params()` - The conventions followed by mlsquare in defining model parameters are mentioned below. This method should
    handle the "flattening" of parameters.
    `get_params()` - Calling this method should simply return the models existing parameters.
    `update_params()` - This method should enable updating the model parameters for an instantiated model.
    `adapter` - This attribute should contain the adapter choice you have made for your algorithm.
    `module_name` - The primal module name(should be a string)
    `name` - Name that you wish the model should be reffered by.
    `version` - If an implementation exists for your algorithm and you wish to improve it by a different
    implementation, make sure you add a meaningful version number.
    `model_params` - The parameters required to compile your model. Conventions to be followed are mentioned below.

-------------------
Notes on conventions
--------------------

1. Currently mlsquare supports keras as the backend for proxy models. The convention we follow is similar to that of
   keras with some minor changes.

2. The parameters should be defined as a dictionary of dictionaries. The first level of dicts should represent the
   the each layer. Each layer should be followed by the index of the layer.

3. Sample parameter - The below sample shows the parameters for a keras model with 2 layer(both hidden and visible),
    .. code-block:: python

    model_params = {'layer_1': {'units': 1, 'activation': 'sigmoid'},
                    'layer_2': {'activation':'softmax'}
                    'optimizer': 'adam',
                    'loss': 'binary_crossentropy'
                    }

---------------------
Sample implementation
---------------------

1. To get started, create a base model

    .. code-block:: python

    class MyBaseModel(GeneralizedLinearModel):
    def create_model(self, **kwargs):
        ## To parse your model from 'flattened' to 'nested'
        model_params = _parse_params(self._model_params, return_as='nested')

        model = Sequential()

        ## Define your model
        model.add(Dense(units=model_params['layer_1']['kernel_dim'],
                        trainable=False, kernel_initializer='random_normal',  # Connect with sklearn_config
                        activation=model_params['layer_1']['activation']))
        model.add(Dense(model_params['layer_2']['units'],
                        activation=model_params['layer_2']['activation']))
        model.compile(optimizer=model_params['optimizer'],
                      loss=model_params['loss'],
                      metrics=['accuracy'])

        return model

    The above class inherits from the existing `GeneralizedLinearModel`. For most use cases, this would be sufficient,
    unless you wish to overwrite the `set_params()` method.


    .. code-block:: python

        @registry.register
        class MyModel(MyBaseModel):
            def __init__(self):
                # Import the adapter
                self.adapter = MyAdapter
                self.module_name = 'PrimalModuleName'
                self.name = 'ModelName'
                self.version = 'default'
                model_params = {'layer_1': {'units': 10,
                                            'activation': 'linear'
                                            },
                                'layer_2': {
                                            'activation': 'softmax'
                                            },
                                'optimizer': 'adam',
                                'loss': 'categorical_hinge'}

                ## Make sure you call this method after the params are defined
                self.set_params(params=model_params, set_by='model_init')

    Note:
        1. Please make sure that you "register" your model in the registery by using the @register.registry decorator.
        2. Define all mandatory attributes mention earlier in your model's `__init__()` method.
        3. Set your params once you have finalized using the `set_params()` method.


----
FAQs
----

1. What do you mean by "transpliling" a model?
    Model transpilation in mlsquare's context refers to converting a given model to it's neural network equivalent.

2. What is a primal model?
    A primal model is model that you wish to transpile to a neural network model.

3. What is a proxy model?
    The proxy model refers to the intermediate state that a primal undergoes to transpile itself to
    a neural network model.

4. What is Registry and what is it used for?
    `mlsquare` maintains a registry of the model mappings defined in the architectures folder. This is to 
    ensure easy access of models at point.