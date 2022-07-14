import tensorflow as tf

class NeuronLayer(tf.keras.layers.Layer):
    ''' Parent class for all neuron layers '''

    def __str__(self):
        return "[" + str(self.n_in) + "->" + str(self.n_out) + "]"

    def __init__(self, n_in, n_out, activation_fn=None,
                 **kwargs):
        
        super(NeuronLayer, self).__init__(input_dim=n_in)
        
        # Number of inputs
        self._n_in  = n_in
        # Number of outpus
        self._n_out = n_out
        # Activation function
        self._activation_fn = activation_fn
        
    @property
    def n_in(self):
        return self._n_in
    
    @property
    def n_out(self):
        return self._n_out
    
    @property
    def activation_fn(self):
        return self._activation_fn
