"""
Setup configuration for Wafer Yield Intelligence System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="wafer-yield-intelligence",
    version="0.1.0",
    author="Rajendar Muddasani",
    author_email="",
    description="Semiconductor wafer yield analysis and retest prediction pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    package_dir={},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Manufacturing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Quality Control",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'flake8>=6.1.0',
            'mypy>=1.5.0',
            'ipykernel>=6.25.0',
        ],
        'docs': [
            'sphinx>=7.1.0',
            'sphinx-rtd-theme>=1.3.0',
        ],
    },
    entry_points={
        # Entry points need implementation before they can work
        # 'console_scripts': [],
    },
    include_package_data=True,
    package_data={
        'stdf': ['*.json'],
        'config': ['*.template'],
    },
    zip_safe=False,
    keywords=[
        'semiconductor',
        'wafer',
        'yield',
        'machine-learning',
        'defect-detection',
        'pattern-recognition',
        'manufacturing',
        'quality-control',
        'STDF',
        'test-data',
    ],
    project_urls={
        'Documentation': '',
        'Source': '',
        'Bug Reports': '',
    },
)
