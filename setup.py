from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="churn_prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements
)
