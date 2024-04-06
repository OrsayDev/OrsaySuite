# OrsaySuite
Data analysis & drift correction in Nionswift using Hyperspy. This repository is architecture around a safe copy between DataItem objects (from Nionswift) to Hyperspy objects. The two-way conversion means that DataItems can be converted to Hyperspy objects, modified using Hyperspy tools, and re-displayed in Nionswift preserving the metadata, calibrations, and traceability.

## Disclaimer
This package is under active development and may have some bugs. We recommend you always have a back up of your data before processing them.
We have tried to make the drift correction as safe as possible. However, as it touches hardware component, you must be an experienced user to use it on your own microscope.
We decline any responsibility in case of hardware hazard.

## Installation

If you want to use the Orsay tools, you need to have [Hyperspy](https://hyperspy.org/) installed (see Hyperspy's website on how to do it).
The current tested version of hyperspy is 1.7.5. As hyperspy and nionswift have sometimes contradictory requirements, we recommend to make a fresh dedicated conda environment.
We also recommend to install nionswift first then hyperspy.

``conda create --name hypernion``

``conda activate hypernion``

``conda install -c nion nionswift nionswift-tool``

``pip install hyperspy==1.7.5``


Clone the OrsaySuite environment and go to the newly created directory:

``git clone https://github.com/OrsayDev/OrsaySuite.git``

``cd OrsaySuite``

Go to your Nionswift conda environment:

``conda activate hypernion``

Install OrsaySuite with ``pip`` by using:

``pip install -e .``

Now, running Nionswift as ``nionswift`` will show you a window called Orsay Tools.



If you want to run the drift correction, you need to have the `nion-instrumentation-kit` installed (be aware of the syntax difference):

``pip install nionswift-instrumentation``

If you are not working on an actual microscope, you may want to test it with a simulated instrument, in which case you should instal:

``pip install nionswift-usim``

## Using OrsayTools

OrsayTools is a standalone application that will appear under "Window" drop-down menu inside Nionswift, as below

![](C:\Users\Lumiere\Downloads\Capture_orsaytools.PNG)

The library is generalized for 1D Hyperspy signals, meaning spectrum or spectrum images.

It is also possible to script any desired Hyperspy functionality. For this, check the examples of this repository:

* DemoScriptAlign.py: Zeroloss alignment
* DemoScriptAxesManager.py: Recovering Hyperspy axes manager from Nionswift DataItem
* DemoScriptDecomposition.py: Signal decomposition (SVD, NMF, etc..)
* DemoScriptDeconvolution.py: Richardson-Lucy deconvolution

## Contributing

Please contact if you wish to contribute to this project.
