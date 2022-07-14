import tensorflow as tf
import numpy as np
from .NeuronLayer import *
from .DenseLayer import *


def softplus_inverse(x):
    ''' Inverse softplus transformation '''
    return x + np.log(-np.expm1(-x))

class RBFLayer(NeuronLayer):
    ''' Radial basis function expansion '''
    
    def __str__(self):
        return "radial_basis_function_layer" + super().__str__()

    def __init__(self, K, cutoff, scope='', dtype=tf.float32, **kwargs):
        
        super(RBFLayer, self).__init__(1, K, None)
        
        with tf.name_scope(scope):
            
            self._K = K
            self._cutoff = cutoff
            
            # Initialize centers (inverse softplus transformation is applied, 
            # such that softplus can be used to guarantee positive values)
            centers = softplus_inverse(np.linspace(1.0,np.exp(-cutoff),K))
            self.centers = tf.nn.softplus(tf.Variable(
                initial_value=np.asarray(centers), trainable=True,
                name="centers", dtype=dtype))
            tf.summary.histogram("rbf_centers", self.centers) 

            # Initialize widths (inverse softplus transformation is applied, 
            # such that softplus can be used to guarantee positive values)
            widths = [softplus_inverse((0.5/((1.0-np.exp(-cutoff))/K))**2)]*K
            self.widths = tf.nn.softplus(tf.Variable(
                initial_value=np.asarray(widths), trainable=True, 
                name="widths", dtype=dtype))
            tf.summary.histogram("rbf_widths", self.widths)
            
    @property
    def K(self):
        return self._K

    @property
    def cutoff(self):
        return self._cutoff
    
    def cutoff_fn(self, D):
        ''' Cutoff function that ensures a smooth cutoff '''
        
        x = D/self.cutoff
        x3 = x**3
        x4 = x3*x
        x5 = x4*x
        return tf.where(x < 1, 1 - 6*x5 + 15*x4 - 10*x3, tf.zeros_like(x))
    
    def __call__(self, D):
        
        D = tf.expand_dims(D, -1) #necessary for proper broadcasting behaviour
        rbf = self.cutoff_fn(D)*(
            tf.exp(-self.widths*(tf.exp(-D) - self.centers)**2))
        
        return rbf



