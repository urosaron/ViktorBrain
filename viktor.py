#!/usr/bin/env python
"""
ViktorBrain System Launcher

This is the main entry point for the ViktorBrain system.
It forwards to the comprehensive system control script.
"""

import sys
import os
import argparse

if __name__ == "__main__":
    # Forward to the system control script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scripts.start_system import main
    
    # Call the main function
    main() 