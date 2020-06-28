from setuptools import setup, find_packages

setup(
    name="neural-datalog-through-time",
    version="1.0",
    packages=find_packages(),
    install_requires=['torch==1.1.0', 'numpy', 'matplotlib', 'pyDatalog', 'sqlalchemy']
)
