import numpy as np

activation_map = {"relu": lambda x: np.max(x, 0), 
                  "linear": lambda x: x,
                  "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
                  "tanh": lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))}

class Neuron:
    """ Defines a simple neuron with certain activation"""
    
    def __init__(self, label="", activation="relu"):
        self.label = label
        self.activation = activation_map[activation]
    
    def __str__(self):
        return f"Neuron: ({self.label}) "
    
    def forward(self, input_val):
        return self.activation(input_val)
    
class Network:
    """ Creates a network with Oja-style hebbian learning"""
    
    def __init__(self, neurons_layer, activation="relu", learning_rate=1e-3):
        self.num_layers = len(neurons_layer)
        self.neurons_layer = neurons_layer
        self.eta = learning_rate
        
        self.network = list()
        for l in range(self.num_layers):
            self.network.append([Neuron(label=f"{l},{i}", activation=activation) for i in range(self.neurons_layer[l])])
        
        self.weights = list()
        for l in range(self.num_layers - 1):
            self.weights.append(np.ones((self.neurons_layer[l], self.neurons_layer[l + 1])))
    
    def __str__(self):
            
        str_out = "Network: \n"
        for i, neurons in enumerate(self.network):
            for neuron in neurons:
                str_out += str(neuron)
            
            
            if i < self.num_layers - 1:
                str_out += f"\nWeight {i} " + str(self.weights[i]) + "\n"
                
        return str_out
                
    def forward(self, input_vals):
        input_activations = np.zeros(self.neurons_layer[0])
        for i, neuron in enumerate(self.network[0]):
            input_activations[i] = neuron.forward(input_vals[i])
    
        layer_act = input_activations
        for l, layer in enumerate(self.network[1:]):
            # cache current activation for learning
            tmp = layer_act.copy()
            
            # Get next layer activations
            layer_act = np.zeros(len(layer))
            for i, neuron in enumerate(layer):
                neuron_input = np.dot(self.weights[l][i, :], tmp)
                layer_act[i] = neuron.forward(neuron_input)
            
            self.update(l, tmp, layer_act)
            
    def update(self, layer, x, y):
        
        for i in range(self.weights[layer].shape[0]):
            for j in range(self.weights[layer].shape[1]):
                
                d_w_ij = self.eta * y[i] * (x[j] - sum([self.weights[layer][k, j] * y[k] for k in range(len(x))]))
                
                self.weights[layer][i, j] += d_w_ij
                
    