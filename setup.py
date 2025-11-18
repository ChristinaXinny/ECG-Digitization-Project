"""Setup script for ECG digitization project."""

import subprocess
import sys
import os

def check_package(package):
    """Check if package is installed."""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def install_package(package):
    """Install package using pip."""
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    """Setup function."""
    print(" ECG Digitization Setup")
    print("=" * 40)

    # Check Python version
    if sys.version_info < (3, 7):
        print(" Python 3.7+ required")
        sys.exit(1)
    else:
        print(f" Python {sys.version.split()[0]} detected")

    # Required packages
    required_packages = [
        'torch',
        'torchvision',
        'timm',
        'numpy',
        'pandas',
        'opencv-python',
        'loguru',
        'tqdm',
        'PyYAML'
    ]

    print("\n Checking required packages...")
    missing_packages = []

    for package in required_packages:
        if package == 'opencv-python':
            import_name = 'cv2'
        elif package == 'PyYAML':
            import_name = 'yaml'
        elif package == 'timm':
            import_name = 'timm'
        else:
            import_name = package

        if check_package(import_name):
            print(f" {package}")
        else:
            print(f" {package} - missing")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n Installing {len(missing_packages)} missing packages...")

        try:
            # Install from requirements file if it exists
            if os.path.exists('requirements_minimal.txt'):
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_minimal.txt"])
                print(" All packages installed from requirements_minimal.txt")
            else:
                # Install individually
                for package in missing_packages:
                    install_package(package)

        except subprocess.CalledProcessError as e:
            print(f" Installation failed: {e}")
            print("\n Try installing manually:")
            print("pip install torch torchvision timm numpy pandas opencv-python loguru tqdm PyYAML")
            return False

    print("\n Setup completed successfully!")
    print("\n You can now start training:")
    print("python simple_train.py")

    return True

if __name__ == "__main__":
    main()