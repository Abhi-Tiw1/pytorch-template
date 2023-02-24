# PyTorch Template - Additions and changes

This repository has been taken from the excellent pytorch template from [this repository](https://github.com/victoresque/pytorch-template).
The purpose of this repository is to add to this template while revising certain important concepts in DL.

## Additional features in the template

Additions made for this template are:
- Regularization: Previously the config file only defined a single loss and no regularization could be performed. I have added regularization (both l1 and l2) into the config file which can be switched on and off using the state variable. The variable *lamba* defines the amount of regularization required and can be used as a hyper-parameter.
- Initialization: Various different initialization methods can now be used. These can include xavier, kaiming or kaiming uniform, to name a few.

## Playing with CNN & LSTM architechtures:

I have tested a Bi-LSTM model for the MNIST problem to familiarize myself with the slightly different structure of it.
This has also caused a change in the config file, where the initial LSTM parameters are also stored.
These are:
- Number of layers
- Hidden Layer size
- Input size
- Number of classes
- Sequence length

After defining these for the LSTM mode, we can reshape the input such that it is now - batch size x sequence x input sequence size
