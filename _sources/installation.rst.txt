============
Installation
============

We recommend using LattiCS in a separate virtual environment to avoid potential issues caused by conflicting package versions.

First, create a new virtual environment::

    $ python -m venv lattics

Activate the environment::

    $ source myenv/bin/activate

On Windows, you may need to use::

    $ lattics\Scripts\activate.bat

Next, install the LattiCS package from the Git repository::

    $ pip install git+https://github.com/kissdaniel/lattics.git

Start a Python shell and verify that the package can be successfully imported::

    $ python
    >>> import lattics

If no error messages appear, the installation was successful.
