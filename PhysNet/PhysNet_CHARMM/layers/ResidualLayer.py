import tensorflow as tf
import numpy as np
from .NeuronLayer import *
from .DenseLayer import *

class ResidualLayer(NeuronLayer):
    
    def __str__(self):
        return "residual_layer"+super().__str__()

    def __init__(self, n_in, n_out, activation_fn=None, 
                 W_init=None, b_init=None, use_bias=True, seed=None, 
                 rate=0.0, scope='', dtype=tf.float32):
        
        super().__init__(n_in, n_out, activation_fn)
        
        self._rate = rate
        
        with tf.name_scope(scope):
            
            self._dense = DenseLayer(
                n_in,  n_out, activation_fn=activation_fn,
                W_init=W_init, b_init=b_init, use_bias=use_bias, 
                seed=seed, scope="dense", dtype=dtype)
            
            self._residual = DenseLayer(
                n_out, n_out, activation_fn=None, 
                W_init=W_init, b_init=b_init, use_bias=use_bias, 
                seed=seed, scope="residual", dtype=dtype)
        
    @property
    def rate(self):
        return self._rate

    @property
    def dense(self):
        return self._dense

    @property
    def residual(self):
        return self._residual

    def __call__(self, x):
        
        #Pre-activation
        if self.activation_fn is not None: 
            y = tf.nn.dropout(self.activation_fn(x), self.rate)
        else:
            y = tf.nn.dropout(x, self.rate)
        
        x += self.residual(self.dense(y))
        
        return x
