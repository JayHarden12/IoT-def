"""
Setup script for the IoT Intrusion Detection System
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="iot-intrusion-detection",
    version="1.0.0",
    author="IoT Security Team",
    author_email="security@iot-detection.com",
    description="Web-based IoT intrusion detection system for home and small office networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iot-security/n-balot-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: System :: Monitoring",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "iot-detection=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="iot, intrusion-detection, cybersecurity, machine-learning, streamlit, n-balot",
    project_urls={
        "Bug Reports": "https://github.com/iot-security/n-balot-detection/issues",
        "Source": "https://github.com/iot-security/n-balot-detection",
        "Documentation": "https://github.com/iot-security/n-balot-detection/wiki",
    },
)
