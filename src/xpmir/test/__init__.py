import os
import pytest

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
RUN_HEAVY_TESTS = os.getenv("XPMIR_RUN_HEAVY_TESTS") == "1"

IN_CONTINUOUS_INTEGRATION = IN_GITHUB_ACTIONS and not RUN_HEAVY_TESTS


def skip_if_ci(*args, **kwargs):
    return pytest.mark.skipif(
        IN_CONTINUOUS_INTEGRATION, reason="Test doesn't run in CI."
    )(*args, **kwargs)
