from setuptools import setup

setup(
    name="OrsaySuite",
    version="0.1.0",
    author="Yves Auad",
    description="Tools for analyzing data using Hyperspy",
    url="https://github.com/yvesauad/yvorsay-instrument",
    packages=['nionswift_plugin.orsay_suite'],
    packages_dir={'nionswift_plugin.orsay_suite': 'src/nionswift_plugin/orsay_suite'},
    package_data={'nionswift_plugin.orsay_suite': [
        'aux_files/config/Gain_Merlin/*',
    ]},
    python_requires='>=3.8.5',
    include_package_data=True,
)