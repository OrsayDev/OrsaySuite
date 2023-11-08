# OrsaySuite
Data analysis & drift correction in Nionswift using Hyperspy

## Installation

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
