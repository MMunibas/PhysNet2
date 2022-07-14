
# PhysNet Model version 2

PhysNet program package for training of the PhysNet model and application in ASE and PyCHARMM

## PhysNet Trainer

to be done

## PhysNet for ASE

to be done

## PhysNet for CHARMM

PhysNet_CHARMM is a modified PhysNet program package for the use in CHARMM via PyCHARMM.

The requirements are the current version of CHARMM with working PyCHARMM 
module, its dependencies and Tensorflow version 2.8 or newer.

Test examples molecular dynamic simualation of a formic acid dimer in water, para-fluorophenol in water and the decomposition reaction of the Criegee molecule can be found in PhysNet/PhysNet_CAHRMM/examples.

## Installation

Download the git repository to your system and add the PhysNet directory to your PYTHONPATH environmental variable:

      export PYTHONPATH={/your/path/to/here}/PhysNet:$PYTHONPATH
