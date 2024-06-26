import sys  # noqa: F401

import pytest  # noqa: F401


def check_test_solver_install(solver_class):
    """Hook called in `test_solver_install`.

    If one solver needs to be skip/xfailed on some
    particular architecture, call pytest.xfail when
    detecting the situation.
    """
    is_platform_macOS = sys.platform == "darwin"
    if is_platform_macOS and solver_class.name == "TabPFN":
        pytest.skip("Running TabPFN on MacOS takes a lot of time.")
