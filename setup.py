from setuptools import setup, find_packages

setup(
    name="iv_surface",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)