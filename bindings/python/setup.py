#!/usr/bin/env python3

from setuptools import setup, Extension
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from glob import glob
import os

# The main interface is through pybind11, but for now we'll use
# the C API directly until we can upgrade to pybind11
ext_modules = [
    Extension(
        'sep_dsl._sep_dsl',
        sources=[
            'ext/sep_dsl/sep_dsl_python.c',
        ],
        include_dirs=[
            '../../src/c_api',  # C API headers
            '../../commercial_package/headers',  # Commercial headers
        ],
        library_dirs=[
            '../../build/lib',  # Built libraries
        ],
        libraries=['sep'],  # Link against libsep.so
        language='c'
    ),
]

setup(
    name="sep-dsl",
    version="1.0.0",
    author="SEP Engine Team",
    author_email="contact@example.com",
    description="SEP DSL - Advanced AGI Pattern Analysis Language",
    long_description="""
SEP DSL is a domain-specific language for advanced AGI pattern analysis.
It provides quantum coherence analysis, entropy measurement, and sophisticated
pattern recognition capabilities through an intuitive DSL syntax.

Features:
- Real-time quantum coherence and entropy analysis
- CUDA-accelerated pattern recognition
- Advanced AGI pattern detection algorithms
- Integration with quantum field harmonics
- Production-grade mathematical validation

Perfect for:
- Scientific computing and research
- IoT sensor data analysis
- Advanced pattern recognition
- AGI and machine learning research
- Real-time signal processing
    """,
    long_description_content_type="text/plain",
    url="https://github.com/SepDynamics/sep-dsl",
    packages=['sep_dsl'],
    package_dir={'sep_dsl': 'lib'},
    ext_modules=ext_modules,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    keywords="dsl pattern-analysis agi quantum-computing machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/SepDynamics/sep-dsl/issues",
        "Source": "https://github.com/SepDynamics/sep-dsl",
        "Documentation": "https://github.com/SepDynamics/sep-dsl#readme",
    },
    zip_safe=False,  # Due to C extension
)
