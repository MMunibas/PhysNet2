import tensorflow as tf
import numpy as np
from .RBFLayer import *
from .NeuronLayer import *
from .DenseLayer import *
from .ResidualLayer import *

class InteractionLayer(NeuronLayer):
    
    def __str__(self):
        return "interaction_layer"+super().__str__()

    def __init__(self, F, K, num_residual, activation_fn=None, seed=None,
                 rate=0.0, dtype=tf.float32, scope='', **kwargs):
        
        super(InteractionLayer, self).__init__(F, K, activation_fn)
        
        with tf.name_scope(scope):
            # Dropout rate
            self._rate = rate
            
            # Transforms radial basis functions to feature space
            self._k2f = DenseLayer(K, F, W_init=tf.zeros([K, F], dtype=dtype),
                                   use_bias=False, seed=seed, scope='k2f',
                                   dtype=dtype)
            
            # Rearrange feature vectors for computing the "message"
            # Central atoms
            self._dense_i = DenseLayer(
                F, F, activation_fn, seed=seed, scope="dense_i", dtype=dtype) 
            # Neighbouring atoms
            self._dense_j = DenseLayer(
                F, F, activation_fn, seed=seed, scope="dense_j", dtype=dtype)
            
            # For performing residual transformation on the "message"
            self._residual_layer = []
            for i in range(num_residual):
                self._residual_layer.append(ResidualLayer(
                    F, F, activation_fn, seed=seed, rate=rate, 
                    scope="residual_layer" + str(i), dtype=dtype))
            
            #For performing the final update to the feature vectors
            self._dense = DenseLayer(
                F, F, seed=seed, scope="dense", dtype=dtype)
            self._u = tf.Variable(
                initial_value=tf.ones([F], dtype=dtype), trainable=True,
                name="u", dtype=dtype)
            tf.summary.histogram("gates",  self.u)  

    @property
    def rate(self):
        return self._rate

    @property
    def k2f(self):
        return self._k2f

    @property
    def dense_i(self):
        return self._dense_i

    @property
    def dense_j(self):
        return self._dense_j

    @property
    def residual_layer(self):
        return self._residual_layer

    @property
    def dense(self):
        return self._dense

    @property
    def u(self):
        return self._u
    
    def __call__(self, x, rbf, idx_i, idx_j):
        
        # Pre-activation
        if self.activation_fn is not None: 
            xa = tf.nn.dropout(self.activation_fn(x), self.rate)
        else:
            xa = tf.nn.dropout(x, self.rate)
        
        # Calculate feature mask from radial basis functions
        g = self.k2f(rbf)
        
        # Calculate contribution of neighbors and central atom
        xi = self.dense_i(xa)
        
        xj = tf.math.segment_sum(g*tf.gather(self.dense_j(xa), idx_j), idx_i)
        
        # Add contributions to get the "message" 
        m = xi + xj 
        for i in range(len(self.residual_layer)):
            m = self.residual_layer[i](m)
        if self.activation_fn is not None: 
            m = self.activation_fn(m)
        
        x = self.u*x + self.dense(m)
        
        return x
