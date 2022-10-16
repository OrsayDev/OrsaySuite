# OrsaySuite
Data analysis in Nionswift using Hyperspy

## Installation

Clone the OrsaySuite environment and go to the newly created directory:

``git clone https://github.com/OrsayDev/OrsaySuite.git``

``cd OrsaySuite``

Go to your Nionswift conda environment:

``conda activate nionswift``

Install OrsaySuite with ``pip`` by using:

``pip install -e .``

Now, running Nionswift as ``nionswift`` will show you a window called Orsay Tools.

## Dependencies

OrsaySuite depends on both Hyperspy and Nionswift
Since both may conflict from time to time, you may want to start on a fresh environment.

``conda create -n hypernion -c nion nionswift nionswift-tool``

Make sure ``nionswift-tool`` is installed, otherwise you may have a conflict with the usage of `Qt5` also used by hyperspy.

Go to your hypernion conda environment:

``conda activate hypernion``

Install Hyperspy

``conda install hyperspy-base -c conda-forge``

Then install OrsaySuite.


## Contributing

Please contact if you wish to contribute to this project.
