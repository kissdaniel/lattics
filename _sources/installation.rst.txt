Installation
============

We recommend using LattiCS in a separate virtual environment to avoid potential issues caused by conflicting package versions.

Install on Linux
----------------

Create a new virtual environment:

.. code-block:: console

	python -m venv lattics

Activate the environment:

.. code-block:: console	

	source lattics/bin/activate

Install the LattiCS package from the Git repository:

.. code-block:: console
	
	pip install git+https://github.com/kissdaniel/lattics.git

Start a Python shell and verify that the package can be successfully imported:

.. code-block:: console

	python
	>>> import lattics

If no error messages appear, the installation was successful.

Install on Windows
------------------

Create a new virtual environment:

.. code-block:: console

	python -m venv lattics

Activate the environment:

.. code-block:: console	

	lattics\Scripts\activate.bat

Install the LattiCS package from the Git repository:

.. code-block:: console
	
	pip install git+https://github.com/kissdaniel/lattics.git

Start a Python shell and verify that the package can be successfully imported:

.. code-block:: console

	python
	>>> import lattics

If no error messages appear, the installation was successful.
