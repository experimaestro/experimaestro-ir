import os
import pytest

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

IN_CONTINUOUS_INTEGRATION = IN_GITHUB_ACTIONS


def skip_if_ci(*args, **kwargs):
    return pytest.mark.skipif(
        IN_CONTINUOUS_INTEGRATION, reason="Test doesn't run in CI."
    )(*args, **kwargs)
