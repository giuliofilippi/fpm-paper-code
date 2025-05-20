# fpm-paper-code

Code for paper: "Lateralised memory networks may explain the use of higher-order visual features in navigating insects".

- LICENSE: GNU general public license

## Description of files

- geometry.py: code used to produce the experienced visual scene from different locations and facing angles in the arena.

- networks.py: contains implementations of neural network classes for modelling the insect Mushroom Bodies. These can be instantiated as a single or a bilaterally organised model.

- generate.py: code used to generate weights for the neural network model instantiations according to different conditions (constant or random).

- functions.py: implementations of basic metrics used to study results of computational experiments and helpertools used to restructure or reorganise data.

- transforms.py: code used to organise the visual input into an array of appropriate dimensions that has undergone the appropriate processing (downsampling, overlap).

- plotting.py: helpertools for plotting the panoramic visual scenes in a coherent format.

- distribution.py: contains the code used to generate a distribution of predicted heading directions (modelling saccade end points) from the novelty sum and difference outputs of a computational model.

- statistical.py: contains the code used to smooth distributions using kernel density estimation, study the location of their major modes and compute their performance.

## Description of folders

- params_fwd: folder for creating and storing parameter values (constant)

- params_rnd: folder for creating and storing parameter values (random)

- writeup_paths: folder containing code used to obtain the sequence of views experienced by a hypothetical ant moving within the arena for the different shapes considered in the paper.

- writeup_shapes: folder used to export svg images of the shapes, annotated with the appropriate angular locations (e.g., FPM).

- writeup_vectors: folder contains the code used to infer and export the distributions and modes from the experiments of Lent et al (2013).

- writeup_notebooks_example: contains an example of the type of computational experiments which constitute the results of the paper.

- writeup_notebooks_base: contains the computational experiments which constitute the main results of the paper.

- writeup_notebooks_perf: contains the analysis of model performance split between different figures and aggregated together.

- writeup_notebooks_sup: contains the supplemental computational experiments such as parameter scans for different shapes and different modes of network weight initialization.

- writeup_exports_: svg exports of the results of computational experiments.
