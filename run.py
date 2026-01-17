#!/usr/bin/env python3
"""
GroundZero AI - Quick Run Script
================================

Usage:
    python run.py              # Interactive chat
    python run.py --dashboard  # Start web dashboard
    python run.py --learn "topic"  # Learn about topic
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from groundzero import GroundZeroAI, main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default: interactive chat
        sys.argv.append("--chat")
    main()
