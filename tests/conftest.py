"""
Pytest configuration file.
Adds project root to Python path to allow imports from main package.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
