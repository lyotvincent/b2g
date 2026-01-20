"""
B2G: Batch-to-Group - Adaptive Batch Grouping for Single-Cell RNA-seq Data
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="b2g_tools",
    version="0.1.3",
    author="B2G Development Team",
    description="Adaptive batch grouping for single-cell RNA-seq data using metacells/leiden clustering and PERMANOVA-based prior selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lyotvincent/b2g",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
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
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "scikit-bio>=0.5.6",
        "scanpy>=1.9.0",
        "anndata>=0.8.0",
        "matplotlib>=3.4.0",
        "metacells>=0.9.0",
        "dynamicTreeCut>=0.1.1",
        "leidenalg>=0.8.0",
    ],
    keywords="single-cell bioinformatics batch-correction clustering metacell leiden",
    project_urls={
        "Bug Reports": "https://github.com/lyotvincent/b2g/issues",
        "Source": "https://github.com/lyotvincent/b2g",
    },
)
