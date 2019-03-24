==========
User Guide
==========

.. py:currentmodule:: mlsquare

This user guide explores the MLSquare API and should provide you with enough information to get you started. Note that this user guide is intended as an introduction to MLSquare, not to Keras or SkLearn or any other packages in general. Readers should already have a basic understanding of the packages they were using and its API.

While the user guide does cover most features, it is not a complete reference
guide. More information about the MLSquare API is available from the :doc:`API
documentation <api>`.

.. contents:: On this page
   :local:

Importing the :py:mod:`mlsquare` module
=======================================

To start using the package, we need to import the module into the python enviroment.

.. code-block:: python

    >>> import mlsquare

If the above command doesn't result in any errors, then the import is successful

.. note::
    To use :py:mod:`mlsquare` you need `Python` 3.6 or higher

Load :py:meth:`dope` method into the enviroment
===============================================

:py:meth:`dope` is the base function, that returns an implementation of a given model to its DNN implementation. Once a model is dope'd, users will be able to use the same work flow as their initial model on the dope'd object.

.. code-block:: python

    >>> from mlsquare.imly import dope

Transpiling an existing model using `dope`
==========================================

To demonstrate :py:meth:`dope`, we will transpile :py:mod:`sklearn` :py:class:`LinearRegression` and use the :py:mod:`sklearn` operations on the transpiled model.

.. code-block:: python

    >>> from sklearn.linear_model import LinearRegression
    >>> model = LinearRegression()
    >>> m = dope(model)

    # Dope maintains the same interface as the base model package
    >>> m.fit(x_train, y_train)
    >>> m.score(x_test, y_test)

.. note::

    :py:meth:`dope` function doesn't support all the packages and the models in the package. A list of supported packages and models is available at the :doc:`Supported Modules and Models <support>`

