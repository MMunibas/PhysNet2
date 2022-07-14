import tensorflow as tf
from .layers.RBFLayer import *
from .layers.InteractionBlock import *
from .layers.OutputBlock      import *
from .activation_fn import *
from .grimme_d3.grimme_d3 import *

import numpy as np

import sys
import ctypes
import pandas
import pycharmm

def softplus_inverse(x):
    '''numerically stable inverse of softplus transform'''
    return x + np.log(-np.expm1(-x))

class PhysNet_Calc(tf.keras.Model):
    def __str__(self):
        return "Neural Network"

    def __init__(self,
                 # Total number of atoms
                 num_atoms,
                 # PhysNet atom indices,
                 ml_indices,
                 # PhysNet atom numbers,
                 ml_Z,
                 # Fluctuating ML charges for ML-MM electrostatic interaction
                 ml_fq,
                 # System atoms charges
                 mlmm_Q,
                 # PhysNet atoms total charge,
                 ml_Q,
                 # Cutoff distance for PhysNet interactions
                 ml_cut,
                 # Cutoff distance for long range interactions 
                 lr_cut,
                 # Cutoff distance for ML/MM electrostatic interactions
                 mlmm_rcut,
                 # Cutoff width for ML/MM electrostatic interactions
                 mlmm_width,
                 # Energy conversion factor
                 econv,
                 # Force conversion factor
                 fconv,
                 # Dimensionality of feature vector
                 F,
                 # Number of radial basis functions
                 K,
                 # Number of building blocks to be stacked
                 num_blocks=3,
                 # Number of residual layers for atomic refinements of 
                 # feature vector
                 num_residual_atomic=2,
                 # Number of residual layers for refinement of message vector
                 num_residual_interaction=2,
                 # Number of residual layers for the output blocks
                 num_residual_output=1,
                 # Adds electrostatic contributions to atomic energy
                 use_electrostatic=True,
                 # Adds dispersion contributions to atomic energy
                 use_dispersion=True,
                 # s6 coefficient for d3 dispersion, by default is learned
                 s6=None,
                 # s8 coefficient for d3 dispersion, by default is learned
                 s8=None,
                 # a1 coefficient for d3 dispersion, by default is learned
                 a1=None,
                 # a2 coefficient for d3 dispersion, by default is learned   
                 a2=None,
                 # Initial value for output energy shift 
                 # (makes convergence faster)
                 Eshift=0.0,
                 # Initial value for output energy scale 
                 # (makes convergence faster)
                 Escale=1.0,
                 # Initial value for output charge shift 
                 Qshift=0.0,
                 # Initial value for output charge scale 
                 Qscale=1.0,
                 # Half (else double counting) of the Coulomb constant 
                 # (default is in units e=1, eV=1, A=1)
                 kehalf=7.199822675975274,
                 # Activation function
                 activation_fn=shifted_softplus, 
                 # Single or double precision
                 dtype=tf.float32,
                 # Random seed
                 seed=None,
                 # Model name
                 name="PhysNet",
                 # Further Keras variables
                 **kwargs):
        
        super(PhysNet_Calc, self).__init__(name=name, **kwargs)
        
        # Total atom number
        self.num_atoms = num_atoms
        
        # ML atoms - atom number
        self.ml_num_atoms = len(ml_indices)
        
        # MM atoms - atom number
        self.mm_num_atoms = self.num_atoms - self.ml_num_atoms
        
        # ML atoms - atom indices
        self.ml_indices = tf.constant(ml_indices, dtype=tf.int32, name="mlidx")
        
        # ML atoms - atom numbers
        self.ml_Z = tf.constant(ml_Z, dtype=tf.int32, name="mlZ")
        
        # ML atoms - fluctuating charges 
        self.ml_fq = ml_fq
        
        # ML&MM atoms - atom charges, but ML charges equal zero
        self.mlmm_Q = tf.constant(mlmm_Q, dtype=dtype, name="mlmm_Qi")
        
        # ML atoms total charge
        self.ml_Q = tf.constant(ml_Q, dtype=dtype, name="mlQ")
        
        # ML atoms - atom indices pointing from MLMM position to ML position
        # 0, 1, 2 ..., ml_num_atoms: ML atom 1, 2, 3 ... ml_num_atoms + 1
        # ml_num_atoms + 1: MM atoms
        ml_idxp = np.full(self.num_atoms, -1)
        for ia, ai in enumerate(self.ml_indices):
            ml_idxp[ai] = ia
        self.ml_idxp = tf.constant(ml_idxp, dtype=tf.int32, name="ml_idxp")
        
        # Electrostatic interaction range
        self._mlmm_rcut = tf.constant(mlmm_rcut, dtype=dtype)
        self._mlmm_rcut2 = tf.constant(mlmm_rcut**2, dtype=dtype)
        self._mlmm_width = tf.constant(mlmm_width, dtype=dtype)
        
        # Energy conversion factor from PhysNet energy unit into kcal/mol
        self.econv = econv
        
        # Energy conversion factor from PhysNet force unit into kcal/mol/A
        self.fconv = fconv
        
        assert(num_blocks > 0)
        self._num_blocks = num_blocks
        self._ftype = dtype
        self._kehalf = kehalf
        self._F = F
        self._K = K
        self._ml_cut = ml_cut #cutoff for ML interactions
        self._lr_cut = lr_cut #cutoff for long-range interactions within ML
        if lr_cut is None:
            self._max_rcut = np.max([ml_cut, mlmm_rcut])
        else:
            self._max_rcut = np.max([ml_cut, lr_cut, mlmm_rcut])
        self._max_rcut2 = self.max_rcut**2
        self._use_electrostatic = use_electrostatic
        self._use_dispersion = use_dispersion
        self._activation_fn = activation_fn
        
        # Probability for dropout regularization
        self._rate = tf.constant(0.0, shape=[], name="rate")

        # Atom embeddings (we go up to Pu(94): 95 - 1 ( for index 0))
        self._embeddings = tf.Variable(
            tf.random.uniform(
                [95, self.F], minval=-np.sqrt(3), maxval=np.sqrt(3), 
                seed=seed, dtype=dtype), 
            trainable=True, name="embeddings", dtype=dtype)
        
        tf.summary.histogram("embeddings", self.embeddings)

        # Radial basis function expansion layer
        self._rbf_layer = RBFLayer(K, ml_cut, scope="rbf_layer")
        
        # Initialize variables for d3 dispersion (the way this is done, 
        # positive values are guaranteed)
        if s6 is None:
            self._s6 = tf.nn.softplus(tf.Variable(
                initial_value=softplus_inverse(d3_s6), trainable=True, 
                name="s6", dtype=dtype))
        else:
            self._s6 = tf.Variable(
                initial_value=s6, trainable=False, 
                name="s6", dtype=dtype)
        tf.summary.scalar("d3-s6", self.s6)
        
        if s8 is None:
            self._s8 = tf.nn.softplus(tf.Variable(
                initial_value=softplus_inverse(d3_s8), trainable=True, 
                name="s8", dtype=dtype))
        else:
            self._s8 = tf.Variable(
                initial_value=s8, trainable=False, 
                name="s8", dtype=dtype)
        tf.summary.scalar("d3-s8", self.s8)
        
        if a1 is None:
            self._a1 = tf.nn.softplus(tf.Variable(
                initial_value=softplus_inverse(d3_a1), trainable=True, 
                name="a1", dtype=dtype))
        else:
            self._a1 = tf.Variable(
                initial_value=a1, trainable=False, 
                name="a1", dtype=dtype)
        tf.summary.scalar("d3-a1", self.a1)
        
        if a2 is None:
            self._a2 = tf.nn.softplus(tf.Variable(
                softplus_inverse(d3_a2), trainable=True, 
                name="a2", dtype=dtype))
        else:
            self._a2 = tf.Variable(
                initial_value=a2, trainable=False, 
                name="a2", dtype=dtype)
        tf.summary.scalar("d3-a2", self.a2)

        # Initialize output scale/shift variables
        self._Eshift = tf.Variable(
            initial_value=tf.constant(Eshift, shape=[95], dtype=dtype),
            name="Eshift", dtype=dtype)
        self._Escale = tf.Variable(
            initial_value=tf.constant(Escale, shape=[95], dtype=dtype),
            name="Escale", dtype=dtype)
        self._Qshift = tf.Variable(
            initial_value=tf.constant(Qshift, shape=[95], dtype=dtype), 
            name="Qshift", dtype=dtype)
        self._Qscale = tf.Variable(
            initial_value=tf.constant(Qscale, shape=[95], dtype=dtype), 
            name="Qscale", dtype=dtype)

        # Embedding blocks and output layers
        self._interaction_block = []
        self._output_block = []
        
        for i in range(num_blocks):
            
            self.interaction_block.append(
                InteractionBlock(
                    F, K, num_residual_atomic, num_residual_interaction,
                    activation_fn=activation_fn, seed=seed, 
                    rate=self.rate, scope="interaction_block" + str(i), 
                    dtype=dtype))
            
            self.output_block.append(
                OutputBlock(
                    F, num_residual_output, activation_fn=activation_fn, 
                    seed=seed, rate=self.rate, scope="output_block" + str(i),
                    dtype=dtype))
                            
        # Save checkpoint to write/read the models variables
        self._saver = tf.train.Checkpoint(model=self)
        
        #self.start_charmm = time.time()
        
    #@tf.function
    def calculate_ml_interatomic_distances(self, R, idxi, idxj):
        ''' Calculate interatomic distances '''
        
        # Gather positions
        Ri = tf.gather(R, idxi)
        Rj = tf.gather(R, idxj)
        
        # Gather atom pairs within max interaction range
        Dij2 = tf.reduce_sum((Ri - Rj)**2, -1)
        idxr = tf.squeeze(tf.where(Dij2 < self.max_rcut2))
        
        # Interacting atom pair distances
        Dij = tf.sqrt(tf.nn.relu(tf.gather(Dij2, idxr)))
        
        # Reduce to interacting atom pairs
        idxi_r = tf.gather(idxi, idxr)
        idxj_r = tf.gather(idxj, idxr)
        
        idxi_z = tf.gather(self.ml_idxp, idxi_r)
        idxj_z = tf.gather(self.ml_idxp, idxj_r)
        
        return Dij, idxi_z, idxj_z
    
    #@tf.function
    def atomic_properties(self, mlmm_R, ml_idxi, ml_idxj):
        ''' Calculates the atomic energies, charges and distances 
            (needed if unscaled charges are wanted e.g. for loss function) '''
        
        with tf.name_scope("atomic_properties"):
        
            # Calculate distances (for long range interaction)
            ml_Dij, ml_idxi_z, ml_idxj_z = \
                self.calculate_ml_interatomic_distances(
                    mlmm_R, ml_idxi, ml_idxj)
            
            # Calculate radial basis function expansion
            ml_rbf = self.rbf_layer(ml_Dij)
            
            # Initialize feature vectors according to embeddings for 
            # nuclear charges
            x = tf.gather(self.embeddings, self.ml_Z)
            
            # Apply blocks
            ml_Ea = 0 #atomic energy 
            ml_Qa = 0 #atomic charge
            
            for i in range(self.num_blocks):
                x = self.interaction_block[i](
                    x, ml_rbf, ml_idxi_z, ml_idxj_z)
                out = self.output_block[i](x)
                ml_Ea = ml_Ea + out[:,0]
                ml_Qa = ml_Qa + out[:,1]
                
            # Apply scaling/shifting
            ml_Ea = (
                tf.gather(self.Escale, self.ml_Z) * ml_Ea 
                + tf.gather(self.Eshift, self.ml_Z))
            ml_Qa = (
                tf.gather(self.Qscale, self.ml_Z) * ml_Qa 
                + tf.gather(self.Qshift, self.ml_Z))
            
        return ml_Ea, ml_Qa, ml_Dij, ml_idxi_z, ml_idxj_z
    
    #@tf.function
    def energy_from_atomic_properties(
        self, ml_Ea, ml_Qa, ml_Dij, ml_idxi_z, ml_idxj_z):
        ''' Calculates the energy given the atomic properties (in order to 
            prevent recomputation if atomic properties are calculated) '''
        
        with tf.name_scope("energy_from_atomic_properties"):
            
            # Scale charges such that they have the desired total charge
            ml_Qa = self.scaled_charges(ml_Qa)
            
            # Add electrostatic and dispersion contribution to atomic energy
            if self.use_electrostatic:
                ml_Ea = ml_Ea + self.electrostatic_energy_per_atom(
                    ml_Dij, ml_Qa, ml_idxi_z, ml_idxj_z)
            if self.use_dispersion:
                if self.lr_cut is not None:   
                    ml_Ea = ml_Ea + d3_autoev*edisp(
                        self.ml_Z, ml_Dij/d3_autoang, ml_idxi_z, ml_idxj_z,
                        s6=self.s6, s8=self.s8, a1=self.a1, a2=self.a2,
                        cutoff=self.lr_cut/d3_autoang)
                else:
                    ml_Ea = ml_Ea + d3_autoev*edisp(
                        self.ml_Z, ml_Dij/d3_autoang, ml_idxi_z, ml_idxj_z,
                        s6=self.s6, s8=self.s8, a1=self.a1, a2=self.a2)
            
        return tf.math.reduce_sum(ml_Ea), ml_Qa
    
    def calculate_mlmm_interatomic_distances(self, R, idxi, idxk, idxp):
        ''' Calculate interatomic distances '''
        
        # Gather positions
        Ri = tf.gather(R, idxi)
        Rk = tf.gather(R, idxk)
        
        # Gather atom pairs within max interaction range
        Dik2 = tf.reduce_sum((Ri - Rk)**2, -1)
        idxr = tf.squeeze(tf.where(Dik2 < self.max_rcut2))
        
        # Interacting atom pair distances
        Dik = tf.sqrt(tf.nn.relu(tf.gather(Dik2, idxr)))
        
        # Reduce to interacting atom pairs
        idxi_r = tf.gather(idxi, idxr)
        idxp_r = tf.gather(idxp, idxr)
        
        return Dik, idxi_r, idxp_r
    
    #@tf.function
    def non_bonded_Eelc(
        self, mlmm_R, ml_Qa, mlmm_idxi, mlmm_idxk, mlmm_idxk_p):
        ''' Calculates the electrostatic interaction between ML atoms in 
            the center cell with all MM atoms in the non-bonded lists'''
        
        # Calculate ML(center) - MM(center,image) atom distances
        mlmm_Dik, mlmm_idxi_r, mlmm_idxp_r = \
                self.calculate_mlmm_interatomic_distances(
                    mlmm_R, mlmm_idxi, mlmm_idxk, mlmm_idxk_p)
        mlmm_idxi_z = tf.gather(self.ml_idxp, mlmm_idxi_r)
        
        # Get ML and MM atom charges
        ml_Qai_r = tf.gather(ml_Qa, mlmm_idxi_z)
        mm_Qak_r = tf.gather(self.mlmm_Q, mlmm_idxp_r)
        
        # Get electrostatic interaction energy
        mlmm_Eele = self.electrostatic_energy_per_atom_to_point_charge(
            mlmm_Dik, ml_Qai_r, mm_Qak_r)
        return mlmm_Eele
    
    def charmm_energy_and_forces(
        self, Natom, Ntrans, Natim, x, y, z, dx, dy, dz, imattr,
        Nmlp, Nmlmmp, idxi, idxj, idxu, idxv, idxp):
        ''' Calculates the total energy and forces (including electrostatic 
            interactions)'''
        
        
        #end_charmm = time.time()
        #print("CHARMM stuff: {:.6f} s".format(end_charmm - self.start_charmm))
        
        #start1 = time.time()
        
        # Assign all positions 
        if Ntrans:
            mlmm_R = tf.transpose(tf.constant(
                [x[:Natim], y[:Natim], z[:Natim]], 
                shape=(3, Natim), dtype=self.ftype))
        else:
            mlmm_R = tf.transpose(tf.constant(
                [x[:Natom], y[:Natom], z[:Natom]], 
                shape=(3, Natom), dtype=self.ftype))
        ml_idxi = tf.constant(idxi[:Nmlp], shape=(Nmlp), dtype=tf.int32)
        ml_idxj = tf.constant(idxj[:Nmlp], shape=(Nmlp), dtype=tf.int32)
        mlmm_idxi = tf.constant(idxu[:Nmlmmp], shape=(Nmlmmp), dtype=tf.int32)
        mlmm_idxk = tf.constant(idxv[:Nmlmmp], shape=(Nmlmmp), dtype=tf.int32)
        mlmm_idxk_p = tf.constant(idxp[:Nmlmmp], shape=(Nmlmmp), dtype=tf.int32)
        
        #end1 = time.time()
        #print("Assignment Input: {:.6f} s".format(end1 - start1))
        
        #start2 = time.time()
        
        E, F = self.energy_and_forces(
            mlmm_R, ml_idxi, ml_idxj, mlmm_idxi, mlmm_idxk, mlmm_idxk_p)
        
        #print("mlmm_R shape:", mlmm_R.shape)
        #print("Memory usage: {:} (kb)".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        
        #end2 = time.time()
        #print("Computation: {:.6f} s".format(end2 - start2))
        
        #start3 = time.time()
        
        F = F.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        #F = F.numpy().reshape(-1)
        for idxa in range(Natom):
            ii = 3*idxa
            dx[idxa] += F[ii]
            dy[idxa] += F[ii+1]
            dz[idxa] += F[ii+2]
        if Ntrans:
            for ii in range(Natom, Natim):
                idxa = imattr[ii]
                jj = 3*ii
                dx[idxa] += F[jj]
                dy[idxa] += F[jj+1]
                dz[idxa] += F[jj+2]
        
        #end3 = time.time()
        #print("Assignment Output: {:.6f} s".format(end3 - start3))
        
        #self.start_charmm = time.time()
        
        return E.numpy()
        
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=None, dtype=tf.int32),
            tf.TensorSpec(shape=None, dtype=tf.int32),
            tf.TensorSpec(shape=None, dtype=tf.int32),
            tf.TensorSpec(shape=None, dtype=tf.int32),
            tf.TensorSpec(shape=None, dtype=tf.int32)])
    def energy_and_forces(
        self, mlmm_R, ml_idxi, ml_idxj, mlmm_idxi, mlmm_idxk, mlmm_idxk_p):
        
        with tf.name_scope("energy_and_forces"):
            
            with tf.GradientTape() as tape:
                
                tape.watch(mlmm_R)
                
                ml_Ea, ml_Qa, ml_Dij, ml_idxi_z, ml_idxj_z = (
                    self.atomic_properties(
                        mlmm_R, ml_idxi, ml_idxj))
                
                ml_E, ml_Qa  = self.energy_from_atomic_properties(
                    ml_Ea, ml_Qa, ml_Dij, ml_idxi_z, ml_idxj_z)
                
                if self.ml_fq:
                    mlmm_Eele = self.non_bonded_Eelc(
                        mlmm_R, ml_Qa, mlmm_idxi, mlmm_idxk, mlmm_idxk_p)
                else:
                    mlmm_Eele = 0.0
                
                E = ml_E + mlmm_Eele
                
            dEdR = tape.gradient(
                E, mlmm_R)
            
            F = tf.convert_to_tensor(dEdR)
            
            E = E*self.econv
            F = F*self.fconv
            
        return E, F
            
    def scaled_charges(self, Qa):
        ''' Returns scaled charges such that the sum of the partial atomic 
            charges equals Q_tot (defaults to 0) '''
        
        # Return scaled charges (such that they have the desired total charge)
        return Qa + (self.ml_Q - tf.math.reduce_sum(Qa))/self.ml_num_atoms

    def _switch(self, Dij):
        ''' Switch function for electrostatic interaction (switches between
            shielded and unshielded electrostatic interaction) '''
    
        cut = self.ml_cut/2
        x  = Dij/cut
        x3 = x*x*x
        x4 = x3*x
        x5 = x4*x
        
        return tf.where(Dij < cut, 6*x5-15*x4+10*x3, tf.ones_like(Dij))

    def electrostatic_energy_per_atom(self, Dij, Qa, idx_iz, idx_jz):
        ''' Calculates the electrostatic energy per atom for very small 
            distances, the 1/r law is shielded to avoid singularities '''
    
        # Gather charges
        Qi = tf.gather(Qa, idx_iz)
        Qj = tf.gather(Qa, idx_jz)
        
        # Calculate variants of Dij which we need to calculate
        # the various shileded/non-shielded potentials
        DijS = tf.sqrt(Dij*Dij + 1.0) #shielded distance
        
        # Calculate value of switching function
        switch = self._switch(Dij) #normal switch
        cswitch = 1.0-switch #complementary switch
        
        if self.lr_cut is None:
            
            Eele_ordinary = 1.0/Dij
            Eele_shielded = 1.0/DijS
            
            # Combine shielded and ordinary interactions and apply prefactors 
            Eele = self.kehalf*Qi*Qj*(
                cswitch*Eele_shielded + switch*Eele_ordinary)
            
        else:
            
            cut   = self.lr_cut
            cut2  = self.lr_cut*self.lr_cut
            
            Eele_ordinary = 1.0/Dij  +  Dij/cut2 - 2.0/cut
            Eele_shielded = 1.0/DijS + DijS/cut2 - 2.0/cut
            
            # Combine shielded and ordinary interactions and apply prefactors 
            Eele = self.kehalf*Qi*Qj*(
                cswitch*Eele_shielded + switch*Eele_ordinary)
            Eele = tf.where(Dij <= cut, Eele, tf.zeros_like(Eele))
            
        return tf.math.segment_sum(Eele, idx_iz) 
    
    def _cutoff(self, Dmlmm):
        ''' Switch function for electrostatic interaction (switches between
            shielded and unshielded electrostatic interaction) '''
        
        x  = (Dmlmm - self.mlmm_rcut + self.mlmm_width)/self.mlmm_width
        x3 = x*x*x
        x4 = x3*x
        x5 = x4*x
        
        cutoff = tf.where(
            Dmlmm < self.mlmm_rcut, tf.ones_like(Dmlmm), tf.zeros_like(Dmlmm))
        
        cutoff = tf.where(
            tf.logical_and(
                Dmlmm > self.mlmm_rcut - self.mlmm_width, Dmlmm < self.mlmm_rcut), 
            1-6*x5+15*x4-10*x3, cutoff)
        
        return cutoff

    def electrostatic_energy_per_atom_to_point_charge(
        self, Dik, Qai, Qak):
        ''' Calculate electrostatic interaction between QM atom charge and 
            MM point charge based on shifted Coulomb potential scheme'''
        
        # Cutoff weighted reciprocal distance
        cutoff = self._cutoff(Dik)
        rec_d = cutoff/Dik
        
        # Shifted Coulomb energy
        QQ = 2.0*self.kehalf*Qai*Qak
        Eele = QQ/Dik - QQ/self.mlmm_rcut*(2.0 - Dik/self.mlmm_rcut)
        
        return tf.math.reduce_sum(cutoff*Eele) 

    def save(self, save_file):
        ''' Save the current model '''
        self.saver.write(save_file)
        
    def restore(self, load_file):
        ''' Load a model '''
        self.saver.read(load_file).assert_consumed()
    
    def restore_v1(self, load_file):
        ''' Load a model v1 '''
        
        checkpoint = tf.train.load_checkpoint(load_file)
        chkvars = tf.train.list_variables(load_file)
        chkvarnames = [var[0] for var in chkvars]
        
        ii = 0
        iiall = len(chkvarnames)
        for var in self.trainable_variables:
            
            varname = var.name
            var.assign(checkpoint.get_tensor(varname))
            
            if varname in chkvarnames:
                #print(True)
                ii += 1
                chkvarnames.remove(varname)
           
        varname = "rbf_layer/centers:0"
        if varname in chkvarnames:
            
            self.rbf_layer.centers = tf.nn.softplus(
                checkpoint.get_tensor(varname))
            ii += 1
            chkvarnames.remove(varname)
                
        varname = "rbf_layer/widths:0"
        if varname in chkvarnames:
            
            self.rbf_layer.widths = tf.nn.softplus(
                checkpoint.get_tensor(varname))
            ii += 1
            chkvarnames.remove(varname)
        
        #print(ii, "/", iiall)
        #print("Not loaded:", chkvarnames)
        
        
        
    
    @property
    def rate(self):
        return self._rate
    
    @property
    def num_blocks(self):
        return self._num_blocks

    @property
    def ftype(self):
        return self._ftype

    @property
    def saver(self):
        return self._saver

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def Eshift(self):
        return self._Eshift

    @property
    def Escale(self):
        return self._Escale
  
    @property
    def Qshift(self):
        return self._Qshift

    @property
    def Qscale(self):
        return self._Qscale

    @property
    def s6(self):
        return self._s6

    @property
    def s8(self):
        return self._s8
    
    @property
    def a1(self):
        return self._a1

    @property
    def a2(self):
        return self._a2

    @property
    def use_electrostatic(self):
        return self._use_electrostatic

    @property
    def use_dispersion(self):
        return self._use_dispersion

    @property
    def use_mlmm(self):
        return self._use_mlmm

    @property
    def mlmm_rcut(self):
        return self._mlmm_rcut

    @property
    def mlmm_rcut2(self):
        return self.mlmm_rcut2

    @property
    def mlmm_width(self):
        return self._mlmm_width

    @property
    def cell(self):
        return self._cell

    @property
    def kehalf(self):
        return self._kehalf

    @property
    def F(self):
        return self._F

    @property
    def K(self):
        return self._K

    @property
    def ml_cut(self):
        return self._ml_cut

    @property
    def lr_cut(self):
        return self._lr_cut
    
    @property
    def max_rcut(self):
        return self._max_rcut
    
    @property
    def max_rcut2(self):
        return self._max_rcut2
    
    @property
    def activation_fn(self):
        return self._activation_fn
    
    @property
    def rbf_layer(self):
        return self._rbf_layer

    @property
    def interaction_block(self):
        return self._interaction_block

    @property
    def output_block(self):
        return self._output_block
    
