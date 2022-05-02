from setuptools import setup

setup(
    name="OrsaySuite",
    version="0.0.20",
    author="Yves Auad",
    description="Tools for analyzing data using Hyperspy",
    url="https://github.com/yvesauad/yvorsay-instrument",
    packages=['nionswift_plugin.orsay_suite'],
    packages_dir={'nionswift_plugin.orsay_suite': 'src/nionswift_plugin/orsay_suite'},
    package_data={'nionswift_plugin.orsay_suite': [
        'aux_files/config/Gain_Merlin/60kV-06-09-2021/*',
        'aux_files/config/Gain_Merlin/100kV-12-07-2021/*',
        'aux_files/config/Gain_Merlin/200kV-29-03-2022/*'
    ]},
    python_requires='>=3.8.5',
    include_package_data=True,
)