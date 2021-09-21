from setuptools import setup

setup(
    name = "pointnet-explainability",
    version = "1.0",
    author = "Simone Antonelli",
    scripts=[
        "scripts/train_pointnet.py",
        "scripts/train_aae.py",
        "scripts/optimize.py",
    ],
)