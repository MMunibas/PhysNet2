import tensorflow as tf
from .NeuronLayer      import *
from .InteractionLayer import *
from .ResidualLayer    import *

class InteractionBlock(NeuronLayer):
    
    def __str__(self):
        return "interaction_block"+super().__str__()

    def __init__(self, F, K, num_residual_atomic, num_residual_interaction,
                 activation_fn=None, seed=None, rate=0.0, scope='', 
                 dtype=tf.float32, **kwargs):
        
        super(InteractionBlock, self).__init__(F, K, activation_fn)
        
        with tf.name_scope(scope):
            
            # Interaction layer
            self._interaction = InteractionLayer(
                F, K, num_residual_interaction, activation_fn=activation_fn, 
                seed=seed, rate=rate, scope="interaction_layer", dtype=dtype)

            # Residual layers
            self._residual_layer = []
            for i in range(num_residual_atomic):
                self._residual_layer.append(ResidualLayer(
                    F, F, activation_fn, seed=seed, rate=rate, 
                    scope="residual_layer" + str(i), dtype=dtype))

    @property
    def interaction(self):
        return self._interaction
    
    @property
    def residual_layer(self):
        return self._residual_layer

    def __call__(self, x, rbf, idx_i, idx_j):
        
        x = self.interaction(x, rbf, idx_i, idx_j)
        
        for i in range(len(self.residual_layer)):
            x = self.residual_layer[i](x)
        
        return x
