#!/usr/bin/env python3
"""
EdukaAI Studio Entry Point

This script serves as the entry point for running EdukaAI Studio.
It ensures the src directory is in the Python path and launches the main application.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Import and run main application
from edukaai_studio.main_simplified import main

if __name__ == "__main__":
    main()
