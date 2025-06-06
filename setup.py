"""
To Setup.
"""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="narrative-agent",
    version="0.1.0",
    author="Edward Chen",
    author_email="ecncms@gmail.com",
    description="Narratives Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ruitao-edward-chen/narrative_agent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "requests>=2.25.0",
        "json5>=0.9.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ],
        "examples": [
            "matplotlib>=3.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "narrative-agent-backtest=examples.backtest_example:run_backtest",
        ],
    },
)
