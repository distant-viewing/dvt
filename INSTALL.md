## Installation

The distant viewing toolkit requires Python 3.7+. It is primarily built and
tested using Anaconda Python, which we suggest installing from here:
[Anaconda Distribution](https://www.anaconda.com/distribution/#download-section).
The package can then be installed through PyPI:

```
pip install dvt
```

Additional Python requirements should be installed automatically through PyPI.
It should also be possible to build from a clean virtual environment under
Python 3.7+.

## Testing

The distant viewing toolkit includes a full test suite, based on the python
modules pytest. To run the tests clone the repository and run the following
from a terminal. These commands install pytest and other optional dependencies,
run the pytest unit tests, and display the coverage statistics:

```
pip install -U pytest
pip install .\[tests,optional\]
cd tests; pytest --disable-warnings --cov=dvt --cov-report term-missing .
```

To properly test the entire toolkit, we need to load and run several
configurations of large neural network] models. Therefore, testing can take
upwards of 20 minutes to complete.
