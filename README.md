# OrsaySuite
Data analysis & drift correction in Nionswift using Hyperspy

## Disclaimer
This package is under active development and may have some bugs. We recommend you always have a back up of your data before processing them.
We have tried to make the drift correction as safe as possible. However, as it touches hardware component, you must be an experienced user to use it on your own microscope.
We decline any responsibility in case of hardware hazard.

## Installation

If you want to use the Orsay tools, you need to have [Hyperspy](https://hyperspy.org/) installed (see Hyperspy's website on how to do it).
As hyperspy and nionswift have sometimes contradictory requirements, we recommend to make a fresh dedicated conda environment.
We also recommend to install nionswift first then hyperspy.

``conda create --name hypernion``

``conda activate hypernion``

``conda install -c nion nionswift nionswift-tool``

``pip install hyperspy``


Clone the OrsaySuite environment and go to the newly created directory:

``git clone https://github.com/OrsayDev/OrsaySuite.git``

``cd OrsaySuite``

Go to your Nionswift conda environment:

``conda activate nionswift``

Install OrsaySuite with ``pip`` by using:

``pip install -e .``

Now, running Nionswift as ``nionswift`` will show you a window called Orsay Tools.



If you want to run the drift correction, you need to have the `nion-instrumentation-kit` installed (be aware of the syntax difference):

``pip install nionswift-instrumentation``

If you are not working on an actual microscope, you may want to test it with a simulated instrument, in which case you should instal:

``pip install nionswift-usim``

## Contributing

Please contact if you wish to contribute to this project.
