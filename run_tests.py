"""
PINNthermo Test Runner
"""

import sys
import os
import pytest

def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    ret = pytest.main(["-v", os.path.join(os.path.dirname(__file__), "tests")])
    sys.exit(ret)

if __name__ == "__main__":
    main()
