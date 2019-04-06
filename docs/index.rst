.. .. image:: _gfx/logo.png
..    :height: 300px
..    :width: 512px
..    :scale: 60 %
..    :alt: MLSquare Logo
..    :align: center

.. |

========
MLSquare
========

**MLSquare** is an open source developer-friendly Python__ library, designed to make use of Deep Learning for Machine Learning developers.

__ http://python.org/

.. note::
    :py:mod:`mlsquare` python library is developed and maintained by `MLSquare Foundation`__

    __ http://mlsquare.org

In the first version we come up with **Interoperable Machine Learning [IMLY]**. `IMLY` is aimed to provide every Machine Learning Algorithm with an equivalent DNN Implementation.

Getting Started!
================

Setting up :py:mod:`mlsquare` is simple and easy

    1. Create a Virtual Environment

    .. code-block:: bash

        virtualenv ~/.venv
        source ~/.venv/bin/activate

    2. Install :py:mod:`mlsquare` package

    .. code-block:: bash

        pip install mlsquare

    3. Import :py:meth:`dope` function from :py:mod:`mlsquare` and pass the :py:mod:`sklearn` model object.

    .. code-block:: python

        >>> from mlsquare.imly import dope
        >>> from sklearn.linear_model import LinearRegression
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.model_selection import train_test_split
        >>> import pandas as pd

        >>> model = LinearRegression()
        >>> data = pd.read_csv('./datasets/diabetes.csv', delimiter=",",
                       header=None, index_col=False)
        >>> sc = StandardScaler()
        >>> data = sc.fit_transform(data)
        >>> data = pd.DataFrame(data)

        >>> X = data.iloc[:, :-1]
        >>> Y = data.iloc[:, -1]
        >>> x_train, x_test, y_train, y_test =
            train_test_split(X, Y, test_size=0.60, random_state=0)
        >>> m = dope(model)

        >>> # All sklearn operations can be performed on m, except that the underlying implementation uses DNN
        >>> m.fit(x_train, y_train)
        >>> m.score(x_test, y_test)

.. note::
    For a comprehensive tutorial please do checkout this `link`__

    __ https://github.com/mlsquare/mlsquare/blob/master/examples/imly.ipynb

Contents
========

.. toctree::
   :maxdepth: 2

   Installation <installation>
   User Guide <user>
   Developer Guide <developer>
   Module Reference <api>
   License <license>
   Authors <authors>
   Issues <issues>
   Changelog <changelog>
   Supported Modules <support>

External links
==============

* `Online documentation <https://mlsquare.readthedocs.io/>`_ (Read the Docs)
* `Downloads <http://pypi.python.org/pypi/mlsquare/>`_ (PyPI)
* `Source code <https://github.com/mlsquare/mlsquare>`_ (Github)


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

.. * :ref:`modindex`

.. _toctree: http://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: http://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: http://sphinx-doc.org/domains.html#the-python-domain
.. _Sphinx: http://www.sphinx-doc.org/
.. _Python: http://docs.python.org/
.. _Numpy: http://docs.scipy.org/doc/numpy
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: http://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: http://scikit-learn.org/stable
.. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
.. _Google style: https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: http://www.sphinx-doc.org/en/stable/domains.html#info-field-lists

.. vim: set spell spelllang=en: