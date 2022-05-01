from setuptools import setup

setup(
    name="OrsaySuite",
    version="0.0.14",
    author="Yves Auad",
    description="Tools for analyzing data using Hyperspy",
    url="https://github.com/yvesauad/yvorsay-instrument",
    packages=['nionswift_plugin'],
    python_requires='>=3.8.5',
    install_requires=["hyperspy==1.6.5"],
    data_files=[('nionswift_plugin/aux_files/config', [
        'nionswift_plugin/aux_files/config/Gain_Merlin/100kV/100kV_Thresh_20.hpsy',
        'nionswift_plugin/aux_files/config/Gain_Merlin/100kV/100kV_Thresh_30.hpsy',
        'nionswift_plugin/aux_files/config/Gain_Merlin/100kV/100kV_Thresh_40.hpsy',
        'nionswift_plugin/aux_files/config/Gain_Merlin/100kV/100kV_Thresh_50.hpsy'
    ])]
)