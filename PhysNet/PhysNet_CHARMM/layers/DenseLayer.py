import tensorflow as tf
import numpy as np
from .NeuronLayer import *
from .util import *

class DenseLayer(NeuronLayer):
    
    def __str__(self):
        return "dense"+super().__str__()

    def __init__(self, n_in, n_out, activation_fn=None, 
                 W_init=None, b_init=None, use_bias=True, 
                 regularization=True, seed=None, scope="", 
                 dtype=tf.float32, **kwargs):
        
        super(DenseLayer, self).__init__(n_in, n_out, activation_fn)
        
        with tf.name_scope(scope):
            
            # Define weight
            if W_init is None:
                W_init = semi_orthogonal_glorot_weights(n_in, n_out, seed=seed) 
            
            self._W  = tf.Variable(
                initial_value=W_init, trainable=True, name="W", dtype=dtype)  
            
            tf.summary.histogram("weights", self.W)        

            # Define l2 loss term for regularization
            if regularization:
                self._l2loss = tf.nn.l2_loss(self.W, name="l2loss")
                tf.compat.v1.add_to_collection("l2loss", self._l2loss)
            else:
                self._l2loss = 0.0

            # Define bias
            self._use_bias = use_bias
            if self.use_bias:
                
                if b_init is None:
                    b_init = tf.zeros([n_out], name="b_init", dtype=dtype)
                
                self._b = tf.Variable(
                    initial_value=b_init, trainable=True, 
                    name="b", dtype=dtype)
                
                tf.summary.histogram("biases",  self.b)       

    @property
    def W(self):
        return self._W

    @property
    def b(self):
        return self._b

    @property
    def l2loss(self):
        return self._l2loss
    
    @property
    def use_bias(self):
        return self._use_bias

    def __call__(self, x):
        
        y = tf.matmul(x, self.W)
        
        if self.use_bias:
            y += self.b
        
        if self.activation_fn is not None: 
            y = self.activation_fn(y)
        
        return y
