Installation
============

Requirements
------------

- Python >= 3.10
- numpy >= 1.23
- scipy >= 1.9
- pandas >= 1.5
- anndata >= 0.8
- statsmodels >= 0.14
- patsy >= 0.5 (used for formula parsing)

From PyPI
---------

pylimma will be available on PyPI from its first tagged release:

.. code-block:: bash

   pip install pylimma

Optional extras:

.. code-block:: bash

   pip install pylimma[plot]   # matplotlib for diagnostic plots
   pip install pylimma[dev]    # pytest, ruff, mypy for running tests
   pip install pylimma[docs]   # sphinx + nbsphinx for building docs

From source
-----------

.. code-block:: bash

   git clone <repo-url>
   cd pylimma
   pip install -e .[dev,plot]

R-parity testing
----------------

pylimma's test suite is fixture-based - the CSV files in
``tests/fixtures/`` are pre-computed by R limma and checked into the
repository. Running ``pytest`` does not require an R installation.

Regenerating the fixtures (only needed if porting a new function or
pulling a newer R limma version) does require R, Bioconductor limma,
and the helper packages listed in :doc:`validation/fixtures`.
