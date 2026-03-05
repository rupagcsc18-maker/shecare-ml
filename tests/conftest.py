# conftest.py
import sys
import os

# Add parent folder to Python path so tests can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))