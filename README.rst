Benchmark for classification methods
====================================
|Build Status| |Python 3.6+|

.. warning::
    This benchmark is under development and it only run with a dev version of
    benchopt, from this PR: https://github.com/benchopt/benchopt/pull/511


Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to **tabular classification methods**:


$$\\min_{w} f(X, w)$$


where $n$ (or ``n_samples``) stands for the number of samples, $p$ (or ``n_features``) stands for the number of features and


$$X \\in \\mathbb{R}^{n \\times p} \\ , \\quad w \\in \\mathbb{R}^p$$


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/tomMoral/benchmark_classification
   $ benchopt run benchmark_classification

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_classification -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/tomMoral/benchmark_classification/workflows/Tests/badge.svg
   :target: https://github.com/tomMoral/benchmark_classification/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
