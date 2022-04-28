from setuptools import setup
from glob import glob

correction_files = glob('nionswift_plugin/orsay_suite/aux_files/config/Gain_Merlin/*/*')

setup(
    name="OrsaySuite",
    version="0.0.16",
    author="Yves Auad",
    description="Tools for analyzing data using Hyperspy",
    url="https://github.com/yvesauad/yvorsay-instrument",
    packages=['nionswift_plugin.orsay_suite'],
    python_requires='>=3.8.5',
    include_package_data=True,
    data_files=[('', correction_files)],
)