========
[ML]² :  Machine Learning for Machine Learning
========

.. image:: https://circleci.com/gh/mlsquare/mlsquare/tree/dev.svg?style=svg
    :target: https://circleci.com/gh/mlsquare/mlsquare/tree/dev

.. image:: https://api.codacy.com/project/badge/Grade/5b23c72bf17246e6b3df610a798f8935
    :target: https://www.codacy.com/app/shakkeel1330/mlsquare?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mlsquare/mlsquare&amp;utm_campaign=Badge_Grade

.. image:: https://api.codacy.com/project/badge/Coverage/5b23c72bf17246e6b3df610a798f8935
    :target: https://www.codacy.com/app/shakkeel1330/mlsquare?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mlsquare/mlsquare&amp;utm_campaign=Badge_Coverage

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/mlsquare/mlsquare/master

ML Square is python library that utilises deep learning techniques to enable interoperability 
between existing standard machine learning frameworks.

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
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.model_selection import train_test_split
        >>> import pandas as pd
        >>> from sklearn.datasets import load_diabetes

        >>> model = LinearRegression()
        >>> diabetes = load_diabetes()

        >>> X = diabetes.data
        >>> sc = StandardScaler()
        >>> X = sc.fit_transform(X)
        >>> Y = diabetes.target
        >>> x_train, x_test, y_train, y_test =
            train_test_split(X, Y, test_size=0.60, random_state=0)

        >>> m = dope(model)

        >>> # All sklearn operations can be performed on m, except that the underlying implementation uses DNN
        >>> m.fit(x_train, y_train)
        >>> m.score(x_test, y_test)

========
Tutorial
========

For a comprehensive tutorial please do checkout this `link`__

__ https://github.com/mlsquare/mlsquare/blob/master/examples/imly.ipynb


==========
Contribute
==========

To get started with contributing, refer our devoloper guide `here`__

__https://github.com/mlsquare/mlsquare/blob/master/docs/developer.rst


For detailed documentation refer `documentation`__

__ http://mlsquare.readthedocs.io


We would love to hear your feedback. Drop us a mail at *info@mlsquare.org*