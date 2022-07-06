# FarNet-II
FarNet-II usage example.

FarNet-II is a deep learning model that improves
the detection of activity on the farside of the Sun
using farside phase-shift maps as an input.

The reliability of th model is tested by comparing the
outputs with binary activity masks extracted from 
STEREO EUV images of the farside.

For more details, check Broock, E. G. et al. A&A, 2022. 

This repository contains a production test for FarNet-II:

· 'input' directory contains two inputs, 
each one with a batch of sequences of phase-shift 
maps sections, for dates outside the training set 
used to train the model (training set takes dates 
from April 12, 2011 to November 27, 2012, and January 
2, 2013 to August 18, 2014). Dates on the name are the 
dates of the central element of the sequence of the 
first and last sequence on the file.

· 'masks' contains the associated activity masks, as
a proxy of the reliability of the network.

· 'outputs' contains FarNet-II outputs for the given
inputs.

· 'farside_to_magnetogram.py' is the script that needs
to run in order to produce the outputs.

· 'FarNet-II.py' is the deep learning model.

· 'graphs.py' is a script to display the results.
