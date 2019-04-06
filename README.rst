========
[ML]² :  Machine Learning for Machine Learning
========

|contributors| |activity|

.. |contributors| image:: https://img.shields.io/github/contributors/mlsquare/mlsquare.svg
  :alt: contributors
  :target: https://github.com/mlsquare/mlsquare/graphs/contributors

.. |activity| image:: https://img.shields.io/github/commit-activity/m/mlsquare/mlsquare.svg
  :alt: activity
  :target: https://github.com/mlsquare/mlsquare/pulse

MLSquare is an open source developer-friendly Python library, designed to make use of Deep Learning for Machine Learning developers.


================
Getting Started!
================

Setting up ``mlsquare`` is simple and easy

    1. Create a Virtual Environment

    .. code-block:: bash

        virtualenv ~/.venv
        source ~/.venv/bin/activate

    2. Install ``mlsquare`` package

    .. code-block:: bash

        pip install mlsquare

    3. Import ``dope`` function from ``mlsquare`` and pass the ``sklearn`` model object

    .. code-block:: python

        >>> from mlsquare.imly import dope
        >>> from sklearn.linear_model import LinearRegression

        >>> model = LinearRegression()
        >>> m = dope(model)

        >>> # All sklearn operations can be performed on m, except that the underlying implementation uses DNN
        >>> m.fit(x_train, y_train)
        >>> m.score(x_test, y_test)

For detailed documentation refer `documentation`__

__ http://mlsquare.readthedocs.io


We would love to hear your feedback. Drop us a mail at *info*[at]*mlsquare.org*