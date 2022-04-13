from setuptools import setup

setup(
    name="OrsaySuite",
    version="0.0.5",
    author="Yves Auad",
    description="Tools for analyzing data using Hyperspy",
    url="https://github.com/yvesauad/yvorsay-instrument",
    packages=['nionswift_plugin'],
    python_requires='>=3.8.5',
    install_requires=["hyperspy>=1.6.5"]
)