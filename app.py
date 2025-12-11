import sys
import os

# Add the current directory to python path to ensure imports work
sys.path.append(os.path.dirname(__file__))

# Import the main app from ui package
from ui.app import main

if __name__ == "__main__":
    main()
