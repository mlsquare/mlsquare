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

.. |last_commit| image:: https://img.shields.io/github/last-commit/mlsquare/mlsquare.svg
  :alt: last_commit
  :target: https://github.com/mlsquare/mlsquare/commits/master

.. |size| image:: https://img.shields.io/github/repo-size/mlsquare/mlsquare.svg
  :alt: size


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

================
Tutorial
================

For a comprehensive tutorial please do checkout this `link`__

__ https://github.com/mlsquare/mlsquare/blob/master/examples/imly.ipynb



For detailed documentation refer `documentation`__

__ http://mlsquare.readthedocs.io


We would love to hear your feedback. Drop us a mail at *info*[at]*mlsquare.org*

.. image:: https://api.codacy.com/project/badge/Grade/d2c3adf3241e49309287cd1b20257dc7
   :alt: Codacy Badge
   :target: https://app.codacy.com/app/shakkeel1330/mlsquare?utm_source=github.com&utm_medium=referral&utm_content=mlsquare/mlsquare&utm_campaign=Badge_Grade_Dashboard