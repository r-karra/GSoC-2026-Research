from setuptools import setup, find_packages

setup(
    name="spirit_q_research",
    version="0.1.0",
    author="Your Name",
    description="Researching the intersection of Quantum AI and LLM Sequence Modeling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "jax>=0.7.1",
        "flax>=0.12.4",
        "pennylane>=0.44.0",
        "optax>=0.2.7",
        "einops",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.11",
)