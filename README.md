# NOMA_AMC_MRCNN

This repository contains supplementary material for the paper "**Modulation Classification for Non-orthogonal Multiple Access System using a Modified Residual-CNN**". The paper can be accessed with https://doi.org/10.1109/WCNC55385.2023.10118621

- The MATLAB file contains the code for dataset generation. By changing the values of N and power ratio, all the dataset mentioned in the paper can be generated.

- Initially, I generated signals for power ratio 4 and N = 200 in pickle format. This file can be accessed from [here](https://drive.google.com/file/d/1nFIllsllFieRZaGeZTHFxiiXaF7ht4vQ/view?usp=sharing.).

- There is one typing mistake in the figure of the model. Please change kernel size from (2x4) to (2x8) in second convolution layer. 

- The model weights for all trained models for all combinations of power ratio and N is given. You can load the weights and regenerate all results given in the paper.

- In case, need any help, you can write to me.
