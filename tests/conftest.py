import pytest

from dvt.utils import setup_tensorflow


@pytest.fixture(scope="session")
def run_setup_tensorflow():
    setup_tensorflow()
