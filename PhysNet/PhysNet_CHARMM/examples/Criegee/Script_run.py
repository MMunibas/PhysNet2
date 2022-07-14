# Test Script to import PhysNet as energy function in CHARMM via PyCHARMM

# Basics
import os
import sys
import ctypes
import pandas
import numpy as np

# ASE
from ase import Atoms
from ase import io
import ase.units as units

# PyCHARMM
import pycharmm
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.nbonds as nbonds
import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.image as image
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.write as write
import pycharmm.settings as settings
import pycharmm.lingo as stream
import pycharmm.select as select
import pycharmm.shake as shake
import pycharmm.cons_fix as cons_fix
import pycharmm.cons_harm as cons_harm
from pycharmm.lib import charmm as libcharmm
import pycharmm.lib as lib

# Step 0: Load parameter files
#-----------------------------------------------------------

# Residue and classical force field parameter
rtf_fn = "ch3choo.rtf"
read.rtf(rtf_fn)
prm_fn = "ch3choo.par"
read.prm(prm_fn, flex=True)

settings.set_bomb_level(-2)
settings.set_warn_level(-1)

# Step 1: Generate system
#-----------------------------------------------------------

# Generate new segment 
residue = "LIG"
read.sequence_string('{0:s}'.format(residue))
gen.new_segment(
    seg_name="SYS",
    setup_ic=True)

# Read positions
criegee = io.read("ch3choo.xyz", format='xyz')
pos = criegee.get_positions()
pandas_pos = pandas.DataFrame({'x': pos[:, 0], 'y': pos[:, 1], 'z': pos[:, 2]})

# Set initial positions
coor.set_positions(pandas_pos)

# Print current coordinates
pos = coor.get_positions().to_numpy(dtype=np.float64)

# Step 2: Set CHARMM Properties
#-----------------------------------------------------------

# Non-bonding parameter
dict_nbonds = {
    'atom': True,
    'vdw': True,
    'vswitch': True,
    'cutnb': 14,
    'ctofnb': 12,
    'ctonnb': 10,
    'cutim': 14,
    'lrc': True,
    'inbfrq': -1,
    'imgfrq': -1
    }

nbond = pycharmm.NonBondedScript(**dict_nbonds)
nbond.run()

# Custom energy
energy.show()

# Write pdb and psf files
write.coor_pdb("ch3choo.pdb", title="Criegee")
write.psf_card("ch3choo.psf", title="Criegee")

# Prepare PhysNet input parameter
selection = pycharmm.SelectAtoms(seg_id='SYS')

# Atomic numbers
Z = criegee.get_atomic_numbers()

# Checkpoint files
checkpoint = "./physnet-criegee/models/criegee-103635_b"

# PhysNet config file
config = "./physnet-criegee/config"

# Model units are eV and Angstrom
econv = 1./(units.kcal/units.mol)
fconv = 1./(units.kcal/units.mol)

charge = 0

# Initialize PhysNet calculator
pycharmm.MLpot(
    selection,
    fq=True,
    Z=Z,
    checkpoint=checkpoint,
    config=config,
    charge=charge,
    econv=econv,
    fconv=fconv,
    v1=True)	# Model is trained by PhysNet using tensorflow 1.x

# Custom energy
energy.show()

# Step 3: Minimization
#-----------------------------------------------------------

# Optimization with PhysNet parameter
minimize.run_sd(**{
    'nstep': 400,
    'tolenr': 1e-5,
    'tolgrd': 1e-5})

# Write pdb and psf files
write.coor_pdb("mini_ch3choo.pdb", title="Criegee optimized")

# Step 4: Set initial conditions
#-----------------------------------------------------------

stream.charmm_script("stream initial_conditions.str")

# Step 5: NVE - CHARMM, PhysNet
#-----------------------------------------------------------

if True:
    
    timestep = 0.0001   # 0.1 fs

    res_file = pycharmm.CharmmFile(
        file_name='nve_seed1.res', file_unit=1, formatted=True, read_only=False)
    dcd_file = pycharmm.CharmmFile(
        file_name='nve_seed1.dcd', file_unit=2, formatted=False, read_only=False)
    vcd_file = pycharmm.CharmmFile(
        file_name='nve_seed1.vcd', file_unit=3, formatted=False, read_only=False)

    # Run some dynamics
    dynamics_dict = {
        'leap': False,
        'verlet': True,
        'cpt': False,
        'new': True,
        'langevin': False,
        'timestep': timestep,
        'start': False,
        'restart': False,
        'nstep': 1000*1./timestep,  # 1 ns
        'nsavc': 0.01*1./timestep,
        'nsavv': 0.01*1./timestep,
        'inbfrq':-1,
        'ihbfrq': 50,
        'ilbfrq': 50,
        'imgfrq': 50,
        'ixtfrq': 1000,
        'iunwri': res_file.file_unit,
        'iuncrd': dcd_file.file_unit,
        'iunvel': vcd_file.file_unit,
        'nsavl':  0,  # frequency for saving lambda values in lamda-dynamics
        'iunldm':-1,
        'ilap': -1,
        'ilaf': -1,
        'nprint': 100, # Frequency to write to output
        'iprfrq': 500, # Frequency to calculate averages
        'isvfrq': 1000, # Frequency to save restart file
        'ntrfrq': 0,
        'ihtfrq': 0,
        'ieqfrq': 0,
        'iasors': 1,
        'iasvel': 0,
        'ichecw': 0,
        'iscale': 0,  # scale velocities on a restart
        'scale': 1,  # scaling factor for velocity scaling
        'echeck':-1}

    dyn_nve = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_nve.run()
    
    res_file.close()
    dcd_file.close()
    vcd_file.close()
