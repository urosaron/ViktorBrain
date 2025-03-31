#!/usr/bin/env python
"""
ViktorBrain Project Setup Script
--------------------------------
Creates the necessary directory structure and initializes empty files
for the ViktorBrain project.
"""

import os
import sys
import shutil
from pathlib import Path


def create_directory_structure():
    """Create the basic directory structure for the project."""
    # Define directories to create
    directories = [
        "src",
        "tests",
        "notebooks",
        "data",
        "data/saved_states",
        "demo_results",
        "integration_results"
    ]
    
    # Get project root (script location)
    project_root = Path(__file__).parent.absolute()
    
    # Create directories
    for directory in directories:
        dir_path = project_root / directory
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return project_root


def create_empty_init_files(project_root):
    """Create empty __init__.py files in Python package directories."""
    python_dirs = ["src", "tests"]
    
    for dir_name in python_dirs:
        init_path = project_root / dir_name / "__init__.py"
        with open(init_path, 'w') as f:
            f.write(f'"""ViktorBrain {dir_name} package."""\n')
        print(f"Created: {init_path}")


def create_placeholder_readme(project_root):
    """Create a placeholder README if it doesn't exist."""
    readme_path = project_root / "README.md"
    if not readme_path.exists():
        with open(readme_path, 'w') as f:
            f.write("# ViktorBrain\n\n")
            f.write("A neural organoid simulation designed to integrate with ViktorAI.\n\n")
            f.write("## Setup\n\n")
            f.write("1. Run `pip install -r requirements.txt` to install dependencies\n")
            f.write("2. Explore the notebooks directory for examples\n")
        print(f"Created: {readme_path}")
    else:
        print(f"Skipped: {readme_path} (already exists)")


def create_placeholder_requirements(project_root):
    """Create a placeholder requirements.txt if it doesn't exist."""
    req_path = project_root / "requirements.txt"
    if not req_path.exists():
        with open(req_path, 'w') as f:
            f.write("numpy>=1.20.0\n")
            f.write("matplotlib>=3.5.0\n")
            f.write("networkx>=2.6.3\n")
            f.write("scikit-learn>=1.0.2\n")
            f.write("scipy>=1.7.3\n")
            f.write("requests>=2.27.1\n")
            f.write("pytest>=7.0.0\n")
        print(f"Created: {req_path}")
    else:
        print(f"Skipped: {req_path} (already exists)")


def main():
    """Execute the setup process."""
    print("Setting up ViktorBrain project structure...")
    project_root = create_directory_structure()
    create_empty_init_files(project_root)
    create_placeholder_readme(project_root)
    create_placeholder_requirements(project_root)
    print("\nProject setup complete! You can now start developing ViktorBrain.")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run tests: pytest tests/")
    print("3. Explore the demo: python demo.py")


if __name__ == "__main__":
    main() 