# Manifold Auto-encoder Model

### Introduction

This repository contains an implementation of the manifold auto-encoder model. Training scripts are designed to learn natural transformations in datasets where point pair supervision is not immediately available. Training occurs in two phases. First, a transport operator (TO) training phase where the auto-encoder weights are fixed. Afterwards, a fine-tuning (FT) phase where the auto-encoder and transport operator weights are changed together in an alternating fashion. Additional scripts are included for creating plots and training comparison methods.

### Prerequisites
```
Python 3.6
Pytorch 1.7
Matlab
SciKit-Learn
numpy
matplotlib
```

Place datasets in the `./data` folder. [The CelebA dataset can be found here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

### Training

In order to train the transport operators in the TO phase, use the following command (note more CLI arguments are available and included in the script, but here we include the values that are most important):

```
python src/train_transop_natural.py
				-Z [latent dimension]
	   			-M [dictionary size]
				-s [supervision method: RES, NN]
				-d [dataset: mnist, svhn, fmnist, celeba, celeba64]
				-z [zeta, L1 regularizer]
				-g [gamma, Frobenius norm regularizer]
				-N [Number of training samples]
				-ae [Relative weighting on auto-encoder objective]
				-l [Latent scale variable]
				-p [Use pretrained autoencoder]
```

Here is an example TO run script to train on 64x64 CelebA images. Note that the script will spend a significant amount of time in the beginning finding nearest-neighbor pairs in a ResNet-18 latent space. Afterwards, this will be saved in a file that can be reused. We recommend reducing the number of training samples (to 50,000 or 10,000) for a faster training run. This comes at a cost of worse quality nearest neighbors.

```
python src/train_transop_natural.py -Z 32 -M 40 -s 'RES' -d 'celeba64' -z 15e-1 -g 1e-5 -N 150000 -ae 0.75 -l 2
```

After the TO phase has completed, the same script can be run with additional arguments to enter the FT phase. We empirically find that gamma should be decreased by a factor of 10-50 for stability in the FT phase. To use the script in the FT phase, simply add `-pto` and `--TOfile`, followed by a path to the saved .pt file from the TO phase.

Here is an example FT run script for 64x64 CelebA images.

```
python src/train_transop_natural.py -Z 32 -M 40 -s 'RES' -d 'celeba64' -z 8-1 -g 5e-7 -N 150000 -ae 0.75 -l 2 -pto --TOfile <ENTER PATH TO .PT FILE>
```

# Reproducing Figures

### Figure 1

#### MNIST
First, run the following python command:
```
python src/generate_transop_paths.py -Z 10 -M 16 -d "mnist" -s "VGG" -z 0.1 -g 2e-6 -l 30 -r 1 -st
```

Then, use the following matlab script `plotTransOptImgOrbits.m` with the .mat files generated in `/results/mnist/mnist_M16_z10_zeta0.1_gamma2e-06_test`. Set folderUse to the folder where the results are located (`../results/mnist/mnist_M16_z10_zeta0.1_gamma2e-06_test`).

#### FMNIST
First, run the following python command:
```
python src/generate_transop_paths.py -Z 10 -M 16 -d "fmnist" -s "VGG" -z 0.5 -g 2e-5 -l 30 -r 1 -st
```
Then, use the following matlab script `plotTransOptImgOrbits.m` with the .mat files generated in `/results/fmnist/fmnist_M16_z10_zeta0.5_gamma2e-05_test`. Set folderUse to the folder where the results are located (`../results/fmnist/fmnist_M16_z10_zeta0.5_gamma2e-05_test`)

### Figure 2

Please refer to the interactive Jupyter notebook `celeba_run_analysis.ipynb`

### Figure 3

#### MNIST
Run the following python command:
```
python src/path_estimate_test.py -Z 10 -M 16 -z 0.1 -g 2e-06 -d "mnist" -r 1 -imgFlag 1 -st
```

And use the .mat files generated in `results/mnist/mnist_M16_z10_zeta0.1_gamma2e-06_test` with the script `plotPathTest_final.m`. Set folderUse to the folder where the results are located (`../results/mnist/mnist_M16_z10_zeta0.1_gamma2e-06_test`).

#### FMNIST
Run the following python command:
```
python src/path_estimate_test.py -Z 10 -M 16 -z 0.5 -g 2e-05 -d "fmnist" -r 1 -imgFlag 1 -st
```

And use the .mat files generated in `results/fmnist/fmnist_M16_z10_zeta0.5_gamma2e-05_test` with the script `plotPathTest_final.m`. Set folderUse to the folder where the results are located (`../results/fmnist/fmnist_M16_z10_zeta0.5_gamma2e-05_test`)


### Figure 4

Run the following python command:
```
python src/path_estimate_test.py -Z 10 -M 16 -z 0.1 -g 2e-06 -d "mnist" -r 1 -imgFlag 0 -st
```

And use the .mat files generated in `results/mnist/mnist_M16_z10_zeta0.1_gamma2e-06_test` with the script `computeMeanExtrapProb.m`. Set folderUse to the folder where the results are located (`../results/mnist/mnist_M16_z10_zeta0.1_gamma2e-06_test`).

#### FMNIST
Run the following python command:
```
python src/path_estimate_test.py -Z 10 -M 16 -z 0.5 -g 2e-05 -d "fmnist" -r 1 -imgFlag 0 -st
```

And use the .mat files generated in `results/fmnist/fmnist_M16_z10_zeta0.5_gamma2e-05_test` with the script `computeMeanExtrapProb.m`. Set folderUse to the folder where the results are located (`../results/fmnist/fmnist_M16_z10_zeta0.5_gamma2e-05_test`)


### Figure 5

#### MNIST

First, run the following python command:
```
python src/compute_coeff_scale.py -Z 10 -M 16 -z 0.1 -g 2e-06 -d "mnist" -r 1 -st
```
Afterwards, use `plotCoeffEncTests.m` for plotting with the .mat files generated in `results/mnist/mnist_M16_z10_zeta0.1_gamma2e-06_test`. Set folderUse to the folder where the results are located (`../results/mnist/mnist_M16_z10_zeta0.1_gamma2e-06_test`). To compute the Isomap embeddings, we use the [Isomap function here](https://www.mathworks.com/matlabcentral/fileexchange/62449-isomap-d-n_fcn-n_size-options). Download the Isomap code and add its folder to your path.

#### FMNIST

First, run the following python command:
```
python src/compute_coeff_scale.py -Z 10 -M 16 -z 0.5 -g 2e-05 -d "fmnist" -r 1 -st
```
Afterwards, use `plotCoeffEncTests.m` for plotting with the .mat files generated in `results/fmnist/fmnist_M16_z10_zeta0.5_gamma2e-05_test`. Set folderUse to the folder where the results are located (`../results/fmnist/fmnist_M16_z10_zeta0.5_gamma2e-05_test`). To compute the Isomap embeddings, we use the [Isomap function here](https://www.mathworks.com/matlabcentral/fileexchange/62449-isomap-d-n_fcn-n_size-options). Download the Isomap code and add its folder to your path.




### Figure 6

First, run the following python command:
```
python src/sample_zeta_test.py -Z 10 -M 16 -z 0.1 -g 2e-06 -d "mnist" -r 1 -st
```
Afterwards, use `plotSampledZeta.m` for plotting with the .mat files generated in `results/mnist/mnist_M16_z10_zeta0.1_gamma2e-06_test`. Set folderUse to the folder where the results are located (`../results/mnist/mnist_M16_z10_zeta0.1_gamma2e-06_test`). 

#### FMNIST

First, run the following python command:
```
python src/sample_zeta_test.py -Z 10 -M 16 -z 0.5 -g 2e-05 -d "fmnist" -r 1 -st
```
Afterwards, use `plotSampledZeta.m` for plotting with the .mat files generated in `results/fmnist/fmnist_M16_z10_zeta0.5_gamma2e-05_test`. Set folderUse to the folder where the results are located (`../results/fmnist/fmnist_M16_z10_zeta0.5_gamma2e-05_test`)
