==================
Installation Guide
==================

This guide describes how to install `mlsquare`

.. contents:: On this page
   :local:


Setting up a virtual environment
================================

The recommended way to install :py:mod:`mlsquare` is to use a virtual
environment created by ``virtualenv``. Setup and activate a new virtual
environment like this::

   $ virtualenv envname
   $ source envname/bin/activate

If you use the ``virtualenvwrapper`` scripts, type this instead::

   $ mkvirtualenv envname


Installing the :py:mod:`mlsquare` package
=========================================

The next step is to install :py:mod:`mlsquare`. The easiest way is to use `pip` to fetch
the package from the `Python Package Index <http://pypi.python.org/>`_ (PyPI).
This will also install the dependencies for Python.

::

   (envname) $ pip install mlsquare

.. note::

    Installation via ``pip`` installs the stable version in your environment. To install the developer version checkout the package source from `GitHub <https://github.com/mlsquare/mlsquare>`_ and run ``python setup.py install`` from the directory root. Note that developer version is not stable and there are chances that code will break. If you are not sure about it, we suggest you use the stable version.

Testing the installation
========================

Verify that the packages are installed correctly::

   (envname) $ python -c 'import mlsquare'

If you don't see any errors, the installation was successful. Congratulations!


.. rubric:: Next steps

Now that you successfully installed HappyBase on your machine, continue with
the :doc:`User Guide <user>` to learn how to use it.


.. vim: set spell spelllang=en:
