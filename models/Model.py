import torch
import torch.nn as nn
import torch.nn.functional as F


# The `EnergyNet` class is a neural network model that takes in input data and outputs a single value,
# using a specified number of layers and neurons per layer.
dropout_rate = 0.5
class EnergyNet(nn.Module):
    def __init__(self, dim, neurons, layers, batch_size, quad_features = False):
        """
        The function initializes a neural network with a specified number of dimensions, neurons,
        layers, and batch size.
        
        :param dim: The `dim` parameter represents the dimensionality of the input data. It specifies the number of features or variables in the input data
        :param neurons: The "neurons" parameter represents the number of neurons in each hidden layer of the neural network
        :param layers: The "layers" parameter represents the number of hidden layers in the neural network
        :param batch_size: The batch size is the number of samples that will be propagated through the network at once. It is used to divide the dataset into smaller batches for efficient training
        """
        self.ENABLED_QUADRATIC_FEATURES = quad_features
        super(EnergyNet, self).__init__()
        self.dim = dim
        self.neurons = neurons
        self.layers = layers
        self.batch_size = batch_size

        self.inputDense = nn.Linear(dim + self.ENABLED_QUADRATIC_FEATURES*dim*(dim+1)//2, neurons)
        self.hidden = [ x for _ in range(layers-1) for x in [nn.Linear(neurons, neurons),nn.Dropout(dropout_rate)] ]
        self.hidden = nn.ModuleList(self.hidden)
        self.outputDense = nn.Linear(neurons, 1)

    # x represents our data
    def forward(self, x):
        """
        The forward function takes an input x, applies a series of dense and activation layers, and
        returns the output.
        
        :param x: The parameter `x` represents the input to the neural network. It is passed through the input dense layer, followed by a softplus activation function. Then, it is passed through a series of hidden layers, each followed by a softplus activation function. Finally, the output is obtained by passing through the output layer.
        :return: The output of the forward pass through the neural network model.
        """
        if self.ENABLED_QUADRATIC_FEATURES:
            if x.dim() == 1:
                outer_product = torch.outer(x, x)
                indices = torch.triu_indices(x.size(0), x.size(0), offset=0)
                quadratic_features = outer_product[indices[0], indices[1]]
                x = torch.cat((quadratic_features, x), dim=0)
            else: #batches mess things up a bit. 
                x_expanded = x.unsqueeze(2)  # Shape [20, self.dim, 1]
                x_t_expanded = x.unsqueeze(1)  # Shape [20, 1, self.dim]
                outer_product = x_expanded * x_t_expanded  # Shape [20, self.dim, self.dim]
                mask = torch.triu(torch.ones(self.dim, self.dim, dtype=torch.bool))
                quadratic_features = outer_product[:, mask]
                x = torch.cat((quadratic_features, x), dim=1)
    
        x = self.inputDense(x)
        x = F.softplus(x)
        for i in range(self.layers-1):
            x = self.hidden[i](x)
            x = F.softplus(x)
        output = self.outputDense(x)
        return output


#antisymmetry
class TensorNet(nn.Module):
    def __init__(self, dim, neurons, layers, batch_size):
        """
        The function initializes a neural network with a specified number of dimensions, neurons, layers,
        and batch size, and sets up the necessary linear layers and indices for the network.
        
        :param dim: The "dim" parameter represents the dimensionality of the input data. It specifies the number of features or variables in the input data
        :param neurons: The "neurons" parameter represents the number of neurons in each hidden layer of the neural network
        :param layers: The "layers" parameter represents the number of hidden layers in the neural network
        :param batch_size: The batch_size parameter determines the number of samples that will be processed in each forward pass of the neural network. It represents the number of input data points that will be fed into the network simultaneously
        """
        super(TensorNet, self).__init__()
        self.dim = dim
        self.neurons = neurons
        self.layers = layers
        self.batch_size = batch_size

        self.inputDense = nn.Linear(dim, neurons)
        print("dim", dim)
        self.hidden = [ x for _ in range(layers-1) for x in [nn.Linear(neurons, neurons),nn.Dropout(dropout_rate)] ]
        self.hidden = nn.ModuleList(self.hidden)
        self.outputSize = int(dim*(dim-1)/2)
        self.outputDense = nn.Linear(neurons, self.outputSize) #input, output = number of independent entries of the skew-symmetric L
        tri_i = torch.triu_indices(dim, dim, 1) #gives two rows: The first contains rows indexes of the upper triangle, the second contains the column indexes. One diagonal (the main) is skipped. The row size is the number of independent components of L.
        #print(tri_i)
        self.sym_sing = -1
        batch_i = torch.tensor([i for i in range(batch_size) for _ in range(tri_i.size(1))]) #batch-numbers, add as many numbers as independent components of L
        #print(batch_i)
        tri_rep = tri_i.repeat(1, batch_size) #repeat batch-size times horizontally
        #print(tri_rep)
        self.indices = torch.stack((batch_i, tri_rep[0], tri_rep[1]))
        #print("indices=",self.indices)

    # x represents our data
    def forward(self, x):
        """
        The forward function takes an input tensor, applies a series of operations including dense layers
        and activation functions, and returns an output tensor.
        
        :param x: The parameter `x` is the input to the forward function. 
        :return: the variable "output".
        """

        x = self.inputDense(x)
        x = F.softplus(x)
        for i in range(self.layers-1):
            x = self.hidden[i](x)
            x = F.softplus(x)
        data = self.outputDense(x)
        b_n = data.size(0) if data.dim() > 1 else 1
        #print("b_n = ", b_n)
        z = torch.zeros(b_n, self.dim, self.dim)
        #print(z)
        #print("ravel=", data.ravel())
        z[self.indices[0, :b_n*self.outputSize], 
        self.indices[1, :b_n*self.outputSize],
        self.indices[2, :b_n*self.outputSize]] = data.ravel()
        #print(z)
        output = z + self.sym_sing*z.transpose(1, 2)
        return output
        
        #antisymmetry

class JacVectorNet(nn.Module):
    def __init__(self, dim, neurons, layers, batch_size):
        """
        The above function is the initialization method for a neural network model called JacVectorNet,
        which takes in parameters for the dimensions, number of neurons, number of layers, and batch size,
        and sets up the layers and linear transformations for the model.
        
        :param dim: The "dim" parameter represents the dimensionality of the input data. In this case, it is not explicitly used in the code snippet provided. 
        :param neurons: The "neurons" parameter represents the number of neurons in each hidden layer of the neural network
        :param layers: The "layers" parameter represents the number of hidden layers in the neural network
        :param batch_size: The batch_size parameter determines the number of samples that will be processed in each forward and backward pass during training.
        """
        super(JacVectorNet, self).__init__()
        self.dim = dim
        self.neurons = neurons
        self.layers = layers
        self.batch_size = batch_size

        self.inputDense = nn.Linear(3, neurons)
        self.hidden = [nn.Linear(neurons, neurons)
                       for i in range(layers-1)]
        self.hidden = nn.ModuleList(self.hidden)
        self.multiplier = nn.Linear(neurons, 1)
        self.cassimir   = nn.Linear(neurons, 1)


    # x represents our data
    def forward(self, inp):
        """
        The forward function takes an input, applies a series of operations to it, and returns the
        product of the output and the gradient of the Cassimir term with respect to the input, as well as
        the Cassimir term itself.
        
        :param inp: The `inp` parameter represents the input to the forward method. It is the input data that will be passed through the neural network
        :return: two values: `multi * cass_grad` and `cass`.
        """

        x = self.inputDense(inp)
        x = F.softplus(x)
        for i in range(self.layers-1):
            x = self.hidden[i](x)
            x = F.softplus(x)
        multi = self.multiplier(x)
        cass = self.cassimir(x)
        cass_grad = torch.autograd.grad(torch.sum(cass), inp, only_inputs=True, create_graph=True)[0]

        return multi * cass_grad, cass
    
class TauNet(nn.Module):
    def __init__(self,batch_size):
        super(TauNet, self).__init__()
        self.tauM = nn.Parameter(torch.tensor([5.0]))
        self.tauN = nn.Parameter(torch.tensor([5.0]))
        self.batch_size = batch_size

    def tau(self):
        return (self.tauM,self.tauN) 
    def forward(self, inp):
        return self.tauM*inp[0]#+self.tauN(inp[1])
 