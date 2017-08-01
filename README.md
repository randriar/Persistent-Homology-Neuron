# Persistent-Homology-Neuron
Compute the Persistent Homology/ Barcodes for a specific number of spherical shells for each neuron and the Pairwise bootleneck distance to measure the shape

INSTRUCTIONS:

- Put the neuron(.txt extension) in the 'INPUTS' folder
- Run PersistentHomology_Neuron.m
- Enter the number of spherical shells you want to use
- Enter the dimension of homology you want to compute for each neuron

When running the program, you would get the following outputs to analyse each neuron:
- The parameters used in the experiment:l1,l2,sigma,...
- Spherical Shell with their mathematics data: radii,...
- Each neuron with their barcodes and KDE(Kernel Density Estimator)
- The Pairwise distances between the neurons

NB:You need MatLab for this experiment
