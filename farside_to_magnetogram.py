"""
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
used to train the model.

· 'masks' contains the associated activity masks, as
a proxy of the reliability of the network.

· 'outputs' contains FarNet-II outputs for the given
inputs.

· 'farside_to_magnetogram.py' is the script that needs
to run in order to produce the outputs.

· 'FarNet-II.py' is the deep learning model.

· 'graphs.py' is a script to display the results.
"""

# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import argparse
import FarNet_II as model
import scipy.io as io
import h5py
from astropy.io import fits 
import os
import datetime

# Start time of excecution
print('Start',datetime.datetime.now())

# Checkpoint of parameters
checkpoint = 'checkpoint_20.pth.best'

# Class that manage the network on production
class deep_farside(object):
    def __init__(self, parameters):

        # Atributes
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.input_file = parameters['input']
        self.output_file = parameters['output']
        self.max_batch = parameters['maxbatch']
        self.format = self.input_file.split('.')[-1]
        self.format_out = self.output_file.split('.')[-1]
        self.verbose = parameters['verbose']

        # Optimization for hardware
        torch.backends.cudnn.benchmark = True

        # Verbose
        if (self.verbose):
            print("Input format is {0}".format(self.format))

    # Model initialization
    def init_model(self, checkpoint=None, n_hidden=16, loss_type='dice'):

        # Atributes
        self.loss_type = loss_type
        self.checkpoint = checkpoint

        # Model 
        if (self.loss_type == 'dice'):
            self.model = model.UNet(n_channels=1, n_classes=1,n_seq=11, n_hidden=16,device=self.device,batch=10).to(self.device)

        # Verbose
        if (self.verbose):
            print("=> loading checkpoint {0}.pth.best".format(self.checkpoint))
        
        # Checkpoint
        if (self.cuda):
            checkpoint = torch.load('{0}.pth.best'.format(self.checkpoint))
        else:
            checkpoint = torch.load('{0}.pth.best'.format(self.checkpoint), map_location=lambda storage, loc: storage)
        
        # Uptading of model with checkpoint values
        self.model.load_state_dict(checkpoint['state_dict'])        
            
        # Verbose
        if (self.verbose):
            print("=> loaded checkpoint {0}.pth.best".format(self.checkpoint))

    # Forward pass
    def forward(self):

        # Verbose
        if (self.verbose):
            print("Reading input file with the phases...")

        # Loading of inputs
        if (self.format == 'h5'):
            f = h5py.File(self.input_file, 'r')
            phase = f['phases'][:]
            f.close()

        # Shape
        n_cases, n_phases, nx, ny = phase.shape

        # Checking of shape
        assert (n_phases == 11), "n. phases is 11"

        # Verbose
        if (self.verbose):
            print("Normalizing data...")

        # Elimination of NaN values
        phase = np.nan_to_num(phase)

        # Same normalization values as in training input
        # (big enough training sample to have better 
        # statistics)
        phase -= -0.008417484
        phase /= 0.052406892

        # Elimination of positive values (do not inidicate
        # activity)
        phase[phase>0] = 0.0

        # Model in evaluation
        self.model.eval()

        # Number of batches and remaining inputs
        n_batches = n_cases // self.max_batch
        n_remaining = n_cases % self.max_batch

        # Verbose
        if (self.verbose):
            print(" - Total number of maps : {0}".format(n_cases))
            print(" - Total number of batches/remainder : {0}/{1}".format(n_batches, n_remaining))
            
        # Container of output magnetograms
        magnetograms = np.zeros((n_cases,11,nx,ny))

        # Starting index
        left = 0

        # Verbose
        if (self.verbose):
            print("Predicting magnetograms...")

        # Non gradient mode
        with torch.no_grad():

            # Extraction of outputs for whole batches
            for i in range(n_batches):  
                right = left + self.max_batch
                phases = torch.from_numpy(phase[left:right,:,:,:].astype('float32')).to(self.device)                
                output = self.model(phases)
                magnetograms[left:right,:,:,:] = output.cpu().numpy()[:,:,:,:]
                left += self.max_batch

            # Extraction of outputs for extra inputs
            if (n_remaining != 0):
                right = left + n_remaining
                phases = torch.from_numpy(phase[left:right,:,:,:].astype('float32')).to(self.device)                
                output = self.model(phases)
                magnetograms[left:right,:,:,:] = output.cpu().numpy()[:,:,:,:]
                
        # Verbose
        if (self.verbose):
            print("Saving output file...")

        # Generation of output files
        if (self.format_out == 'h5'):
            f = h5py.File(self.output_file, 'w')
            db = f.create_dataset('magnetogram', shape=magnetograms.shape)
            db[:] = magnetograms
            f.close()

# Input folder
path_inputs = './inputs/'

# Reading of inputs
list_inputs = []
for file in sorted(os.listdir(path_inputs)):
    if file.endswith('.h5'):
        list_inputs.append(file)
    
# Output folder
path_outputs = './outputs/'

# Extraction of outputs
for i in range(len(list_inputs)):
    parser = argparse.ArgumentParser(description='''
        Predict a farside magnetogram from the computed phases.
        The input phases needs to be in a file (HDF5) and should contain
        a single dataset with name `phases` of size [n_cases,11,144,180]
        ''')
    parser.add_argument('-i','--input', help='Input file', default=path_inputs+list_inputs[i])
    parser.add_argument('-o','--output', help='Output file', default=path_outputs+'OUT'+list_inputs[i])
    parser.add_argument('-b','--maxbatch', help='Maximum batch size', default=10)
    parser.add_argument('-v','--verbose', help='Verbose', default=True)
    parsed = vars(parser.parse_args())
    deep_farside_network = deep_farside(parsed)
    deep_farside_network.init_model(checkpoint[:-9], n_hidden=16, loss_type='dice')
    deep_farside_network.forward()

# End time of excecution
print('End',datetime.datetime.now())