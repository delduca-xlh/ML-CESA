"""
Setup script for Financial Planning and Valuation System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="financial-planning-system",
    version="1.0.0",
    author="Lihao Xiao",
    author_email="lx2219.cu@gmail.com",
    description="Financial planning and valuation system without plugs or circularity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/financial-planning-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "yfinance>=0.2.28",
        "pdfplumber>=0.9.0",
        "tensorflow>=2.13.0",
        "scikit-learn>=1.3.0",
        "openpyxl>=3.1.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "sphinx>=7.0.0",
        ],
        "ml": [
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
        ],
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.5.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.23.0",
            "notebook>=6.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "financial-model=financial_planning.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "financial_planning": [
            "data/*.json",
            "data/*.yaml",
            "templates/*.xlsx",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/financial-planning-system/issues",
        "Source": "https://github.com/yourusername/financial-planning-system",
        "Documentation": "https://financial-planning-system.readthedocs.io/",
    },
)
