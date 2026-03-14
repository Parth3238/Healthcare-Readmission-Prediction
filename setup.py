"""
Setup script for Healthcare Readmission Prediction package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="healthcare-readmission-prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine Learning system for predicting hospital readmission risk",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Healthcare-Readmission-Prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "flake8>=6.1.0",
            "black>=23.11.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.27.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-model=src.train_model:main",
            "evaluate-model=src.evaluate:main",
            "compare-models=src.model_comparison:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.pkl", "*.html", "*.css", "*.js"],
    },
    zip_safe=False,
)
