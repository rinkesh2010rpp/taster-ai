import re
from pathlib import Path

import taster_ai


def test_the_package_version_matches_pyproject():
    # The publish workflow checks the release tag against pyproject.toml only.
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert taster_ai.__version__ == re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
