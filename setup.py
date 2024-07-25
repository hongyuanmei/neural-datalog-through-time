from setuptools import setup, find_packages

setup(
    name="neural-datalog-through-time",
    version="1.0",
    packages=find_packages(),
    install_requires=['torch==2.2.0', 'numpy', 'matplotlib', 'pyDatalog', 'sqlalchemy']
)
