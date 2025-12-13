"""
setup_directories.py
Run this FIRST to create all necessary directories
"""

import os

def create_all_directories():
    """Create all necessary project directories."""
    
    directories = [
        'data',
        'models',
        'notebooks',
        'src',
        'tests',
        'app',
        'app/templates',
        'app/static'
    ]
    
    print("=" * 70)
    print("CREATING PROJECT DIRECTORY STRUCTURE")
    print("=" * 70)
    print()
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created: {directory}")
        except Exception as e:
            print(f"❌ Error creating {directory}: {e}")
    
    print()
    print("=" * 70)
    print("✅ ALL DIRECTORIES CREATED SUCCESSFULLY!")
    print("=" * 70)
    print()
    print("You can now run: python run_pipeline.py")
    print()

if __name__ == "__main__":
    create_all_directories()