# Linear-response

/*** Here are the instructions to install and run the code generating the simulations of section 5 in the paper "Linear response for spiking neuronal networks with
unbounded memory"
Bruno Cessac, Ignacio Ampuero,Rodrigo Cofre
March 28 2020***/


Included Files
* LinRepBMS.cpp: Sourcefile of LinearResponse computation routine
* Stimuli.cpp: Sourcefile for input stimulus functions
* CMakeLists.txt: CMake file to compile the LinearResponse with installed pranas build
* BMSPotential.cpp and BMSPotential.h: BMSPotential files to replace in the pranas source files

Instructions for Linux Systems
-- You must first install the pranas software available at https://team.inria.fr/biovision/pranas-software/
-- Replace the files BMSPotential.cpp and BMSPotential.h in your <pranas_sourcefiles_folder>/src/pranasCore folder.
-- Compile pranas again to build the changes
- Replace the CMAKE_PREFIX_PATH in the CMakeLists.txt file with your pranas build folder
-- Run cmake . to create the LinRepBMS build file
- Create the N30 folder (the name of the folder depends on the N selected where N is the number of neurons)
- Make the LinRepBMS file and run with ./LinRepBMS

Linear Response Code parameters

plot : 0/1 Boolean to plot the generated rasters
N : Number of neurons
T : Simulation time in bins
transient : Transient time in bins
OF : Offset time in bins
ew : Exitatory weights used in (k,k+1) and (k,k-1) connections
iw : Inhibitory weights used in (k,k-2) and (k,k+2) connections
sigma : Leak coefficient
unit_threshold : Threshold of the unit
Ie : Constant current for spontaneous activity
A : Starting amplitude of the stimulus
M : Number of iterations to average observables
ic : Observed neuron
i1,i2 : Index of neurons to compute pairwise interactions
n1,n2 : Time indices to compute pairwise interactions
choice : 1/0 Boolean to compute spike rate or pairwise activity

Stimulus parameters are explained with inline comments in LinRepBMS.cpp file
