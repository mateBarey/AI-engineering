import numpy as np
from random import seed

n = 2 #number of inputs
num_hidden_lyr = 2 #number of hidden layers
m = [2,2] #number of nodes in each hidden layer
num_nodes_output= 1 #number of nodes in the output layer



#looop through each layer and randmly initialize weights and biases associated with each node
#notice how we are adding 1 to the number of hidden layers  in order to include output layer ***
def initialize_network(num_inputs,num_hidden_lyr,num_nodes_hidden,num_nodes_output):

    num_node_prev = num_inputs #number of nodes in the previous layer
    network= {} #initialize network an empty dictionary

    for layer in range(num_hidden_lyr +1):

        #determine name of layer  -- basically name which layer
        if layer == num_hidden_lyr:
            layer_name ='output'
            num_nodes = num_nodes_output
        else:
            layer_name= 'layer_{}'.format(layer+1)
            num_nodes = num_nodes_hidden[layer]

        #initialize weights and biases asscoiated with each node int the current layer
        network[layer_name]={}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name]= {
                'weights': np.around(np.random.uniform(size=num_node_prev), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
        num_node_prev=num_nodes
    return network

small_network = initialize_network(5,3,[3,2,3],1)


def compute_weighted_sum(inputs,weights,bias):
    return np.sum(inputs*weights) + bias

np.random.seed(12) #generator saves 12 randoms numbers
#generate 5 inputs
inputs=np.around(np.random.uniform(size=5), decimals=2)

print('The inputs to the network are {}'.format(inputs))

node_weights = small_network['layer_1']['node_1']['weights']
node_bias = small_network['layer_1']['node_1']['bias']
weighted_sum = compute_weighted_sum(inputs,node_weights,node_bias)
print('The weighted suma at the first node in the hidden layer is {}'.format(np.around(weighted_sum[0],decimals=4)))

def node_activation(weighted_sum):
    return 1.0 / (1.0 +np.exp(-1 * weighted_sum))

node_output = node_activation(compute_weighted_sum(inputs, node_weights, node_bias))
print('The output of the first node in the hidden layer is {}'.format(np.around(node_output[0], decimals=4)))


def forward_propagate(network, inputs):
    layer_inputs=list(inputs) #start with the input layer as the input to the first hidden lyr
    for layer in network:

        layer_data = network[layer]

        layer_outputs = []
        for layer_node in layer_data:

            node_data = layer_data[layer_node]

            # compute the weighted sum and the output of each node at the same time
            node_output = node_activation(compute_weighted_sum(layer_inputs,node_data['weights'],node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))

        if layer != 'output':
            print('The outputs of the nodes in the hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))
        layer_inputs = layer_outputs #set the output of this layer to be the input to the next layer
    network_predictions = layer_outputs
    return  network_predictions

predictions = forward_propagate(small_network,inputs)
print('The predicted value by the network for the given input is {}'.format(np.around(predictions[0], decimals=4)))

#initialize network to create our neural network and define its weights and biases
my_network = initialize_network(5,3,[2,3,2],3)

#then for a given point

inputs = np.around(np.random.uniform(size=5),decimals=2)

#compute the network predictions

predictions = forward_propagate(my_network,inputs)
print('The predicted values by the network for the given input are {}'.format(predictions))


my_net1 = initialize_network(8,5,[3,2,2,2,3],3)
inputs1=np.around(np.random.uniform(size=8), decimals=2)
predictions1 = forward_propagate(my_net1,inputs1)
print('The predicted values by the network for the given input are {}'.format(predictions1))

my_net2 = initialize_network(4,3,[3,2,3],2)
inputs2=np.around(np.random.uniform(size=4), decimals=2)
predictions2 = forward_propagate(my_net2,inputs2)
print('The predicted values by the network for the given input are {}'.format(predictions2))

x1 = 0.5
x2 = -.35
w1 = 0.55
w2 = 0.45
b1 = 0.15

z = x1*w1 +x2*w2 + b1
print('The output of the linear combination should be {}'.format(round(z,3)))
output = node_activation(round(z,3))
print('The output of the neuron is {}'.format(round(output,3)))
