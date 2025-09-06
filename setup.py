from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mymicrograd",
    version="0.1.0",
    author="Dheeraj Yadav",
    author_email="",
    description="A mini neural network framework for learning deep learning internals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dheerajyadav/mymicrograd",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.6",
    install_requires=[
        # No external dependencies - pure Python!
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
        ],
    },
    keywords="neural-network, deep-learning, autograd, machine-learning, educational",
    project_urls={
        "Bug Reports": "https://github.com/dheerajyadav/mymicrograd/issues",
        "Source": "https://github.com/dheerajyadav/mymicrograd",
        "Documentation": "https://github.com/dheerajyadav/mymicrograd#readme",
    },
)