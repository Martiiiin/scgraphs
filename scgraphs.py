# -*- coding: utf-8 -*-
"""
@author: Martin Prieto Aguilera
"""
import numpy as np
import utils

class Stage:
    """
    This class holds a list of the available Graphs created so far.
    It also has a reference to the instance of the current Graph being used.
    """
    # Reference to the current graph
    _current_graph = None
    
    # List of all Graphs created so far
    _graphs = list()
    
    
#################################################
    
# Now, we define methods inside our package to access the stage variables
def get_current_graph():
    return Stage._current_graph

def set_current_graph(graph):
    Stage._current_graph = graph
    
def clean_all_graphs():
    Stage._graphs = None
    Stage._current_graph = None
    

#################################################
    
    
class Graph:
    """
    A Graph contains a the list of nodes and the method to run the computations.
    """
    def __init__(self):
        # NOTE: On using sets to store the nodes instead of lists: we need the nodes to be appended in an ordered 
        # fashion, so that we have a mean to easily execute the graph, thats why we use a list instead of a set.
        # We'll transform this list to a set when required, in order to do set operations such as intersection, and then cast
        # the result back to a list.
        self.nodes = list()
        # Now, initialize a variable representing the order od execution for the next added node
        self._next_order = 0
        
    # the order variable is probably not needed...
    def get_next_order(self):
        """
        Returns the next order available for a Node to have.
        """
        self._next_order += 1
        return self._next_order
    
    def run(self):
        """
        Runs the Graph. Runs each node in nodes list sequentially.
        """
        for node in self.nodes:
            node.run()
            
    def add_node(self, node):
        """
        Adds a node to this Graph.
        """
        self.nodes.append(node)
        
        
#################################################
        
        
class Operation:
    """
    An Operation represents a rule to obtain the value held by a certain node.
    Definen an Operation requires to specify an iterabla containing the nodes that will be input to the operation and the operation type.
    """
    
    def __init__(self, optype, input_nodes):
        """
        optype : The type of Operation to apply to the values of the input_nodes
        input_nodes : List-like containing the nodes that will intervene in the operation
        """
        self._optype = optype
        self._input_nodes = input_nodes
    
    def f(self):
        """
        Computes the value associated to the Operation, given the values held by input_nodes.
        This is the method to end up being called when performing forwardprop across the Graph.
        """
        if(self._optype == 'identity'):
            return self._identity()
        if(self._optype == 'matmul'):
            return self._matmul()
        if(self._optype == 'tanh'):
            return self._tanh()
        if(self._optype == 'relu'):
            return self._relu()
        if(self._optype == 'softmax'):
            return self._softmax()
        if(self._optype == 'softmax_xentropy'):
            return self._softmax_xentropy()
        if(self._optype == 'addition'):
            return self._add()
        if(self._optype == 'substraction'):
            return self._substract()
        if(self._optype == 'multiplication'):
            return self._multiply()
        if(self._optype == 'division'):
            return self._divide()
        if(self._optype == 'mse'):
            return self._MSE()
        if(self._optype == 'cross_entropy'):
            return self._cross_entropy()
        if(self._optype == 'sum'):
            return self._sum()
        if(self._optype == 'sqrt'):
            return self._sqrt()
        if(self._optype == 'sqr'):
            return self._sqr()
        
        
    def bprop(self, X, G):
        """
        Backpropagation method for the operation.
        
        INPUT:
        
        X: Variable (node) with respect to which we are differentiating
        G: Gradient on the output of the operation.
        
        
        """
        if(self._optype == 'identity'):
            return self._bprop_identity(G)
        if(self._optype == 'matmul'):
            return self._bprop_matmul(X, G)
        if(self._optype == 'addition'):
            return self._bprop_add(X, G)
        if(self._optype == 'substraction'):
            return self._bprop_substract(X, G)
        if(self._optype == 'multiplication'):
            return self._bprop_multiply(X, G)
        if(self._optype == 'tanh'):
            return self._bprop_tanh(G)
        if(self._optype == 'relu'):
            return self._bprop_relu(G)
        if(self._optype == 'mse'):
            return self._bprop_MSE(X, G)
        if(self._optype == 'softmax'):
            return self._bprop_softmax(G)
        if(self._optype == 'cross_entropy'):
            return self._bprop_cross_entropy(X, G)
        if(self._optype == 'softmax_xentropy'):
            return self._bprop_softmax_xentropy(X, G)
        if(self._optype == 'sum'):
            return self._bprop_sum(G)
        if(self._optype == 'sqrt'):
            return self._bprop_sqrt(G)
        if(self._optype == 'sqr'):
            return self._bprop_sqr(G)
        
        
        
    # _bprop_softmax_xentropy
    # Here we add all the definitions of the functions available to obtain the result of the operation
    def _identity(self):
        """
        The identity operation. Returns the value of the only node in _input_nodes.
        """
        X = self._input_nodes[0].value
        return X
    
    
    def _matmul(self):
        """
        Applies the matmul operation to the values of _input_nodes. Returns the result.
        """
        X = self._input_nodes[0]._value # First argument
        Y = self._input_nodes[1]._value # Second argument
        return X.dot(Y)
    
    def _tanh(self):
        """
        Computes the hyperbolic tangent of the node's value.
        """
        X = self._input_nodes[0]._value # First argument
        return np.tanh(X)
    
    def _relu(self):
        """
        Computes the rectified-linear operation of the node's value.
        """
        X = self._input_nodes[0]._value # First argument
        return utils.relu(X)
    
    def _softmax(self):
        """
        Computes the softmax of the node's value.
        """
        X = self._input_nodes[0]._value # First argument
        return utils.softmax(X)
    
    def _add(self):
        """
        Adds two tensors of the same dimension.
        """
        X = self._input_nodes[0]._value # First argument
        Y = self._input_nodes[1]._value # Second argument
        return X+Y
    
    def _substract(self):
        """
        Substracts tensor Y from tensor X (both having the same dimension).
        """
        X = self._input_nodes[0]._value # First argument
        Y = self._input_nodes[1]._value # Second argument
        return X-Y
    
    def _multiply(self):
        """
        Performs element wise multiplication between tensors X and Y (both having the same dimension).
        """
        X = self._input_nodes[0]._value # First argument
        Y = self._input_nodes[1]._value # Second argument
        return X*Y
    
    def _divide(self):
        """
        Performs element wise division between tensors X and Y (both having the same dimension).
        """
        X = self._input_nodes[0]._value # First argument
        Y = self._input_nodes[1]._value # Second argument
        return X/Y
    
    def _MSE(self):
        """
        Performs the MSE function on operand matrices.
        """
        T = self._input_nodes[0]._value # First argument
        Y = self._input_nodes[1]._value # Second argument
        return utils.MSE(T, Y)
    
    def _cross_entropy(self):
        """
        Performs the cross_entropy function on operand matrices.
        """
        T = self._input_nodes[0]._value # First argument
        Y = self._input_nodes[1]._value # Second argument
        return utils.cross_entropy(T, Y)
    
    
    def _softmax_xentropy(self):
        """
        Perform in one step the softmax and cross-entropy, returning only the latter.
        It is intended to be more stable from bprop's perspective.
        """
        T = self._input_nodes[0]._value # First argument
        Z = self._input_nodes[1]._value # Second argument
        # Define the terms of the summatory
        return utils.softmax_xentropy(T, Z)
    
    def _sum(self):
        """
        Applies the sum operation accross a tensor (adds up all its components). Returns a scalar.
        """
        X = self._input_nodes[0]._value
        # Flatten the tensor into a vector
        X_flat = X.flatten()
        return np.sum(X_flat)
    
    def _sqrt(self):
        """
        Computes the sqrt of the elements of a tensor. Returns a tensor of the same size as 
        """
        X = self._input_nodes[0]._value
        return np.sqrt(X)
    
    def _sqr(self):
        """
        Computes the element-wise square of the tensor to produce the node resulting from the operation.
        """
        X = self._input_nodes[0]._value
        return X**2
    
    
    
    ### Define derivatives of the previous functions ###
    
    def _bprop_identiy(self, G):
        """
        Backprop operation for the identity function.
        
        G: The gradient at the output of the operation.
        """
        return G
    
    def _bprop_matmul(self, Z, G):
        """
        Backprop operation for the matmul function.
        
        Z: The node with respect to we want to differentiate.
        G: The gradient at the output of the operation.
        """
        X = self._input_nodes[0]
        Y = self._input_nodes[1]
        # Derivative with respect to the left argument.
        if(Z is X):
            return G.dot(Y._value.T)/len(G)
        # Derivative with respect to the second argument.
        if(Z is Y):
            return (X._value.T).dot(G)/len(G)
        
    def _bprop_add(self, Z, G):
        """
        Backprop operation for add function.
        Z: The node with respect to we want to differentiate.
        """
        
        # Since Numpy allows us to add a vector to a matrix (as long as the vector's dimension is the same as 
        # either a row or a column of that matrix), we need to aggregate in a way that the returned gradient's
        # dimension matches that of the vector.
        
        X = Z._value
        # Check wether G's dimension matches Z, if not, average across the appropriate axis.
        if(type(G) is np.ndarray):
            if(G.shape == X.shape):
                return G
            if (G.shape[0] == X.shape[0]):
                return np.mean(G, axis = 1)
            if (G.shape[1] == X.shape[1]):
                return np.mean(G, axis = 0)
        return G
    
    def _bprop_substract(self, Z, G):
        """
        Backprop operation for substraction function.
        Z: The node with respect to we want to differentiate.
        """
        X = self._input_nodes[0]
        Y = self._input_nodes[1]
        # Derivative with respect to the argument in the left.
        if(Z is X):
            return G
        # Derivative with respect to the argument in the right.
        if(Z is Y):
            return -1*G
        
    def _bprop_tanh(self, G):
        """
        Backprop operation for tanh function.
        Z: The node with respect to we want to differentiate.
        """
        X = self._input_nodes[0]
        return G*(1-(np.tanh(X._value))**2)
    
    def _bprop_relu(self, G):
        """
        Backprop operation for relu function.
        Z: The node with respect to we want to differentiate.
        """
        X = self._input_nodes[0]
        return G*np.where(X._value > 0, 1, 0)

    def _bprop_multiply(self, Z, G):
        """
        Backprop operation for element-wise multiplication function.
        Z: The node with respect to we want to differentiate.
        """
        X = self._input_nodes[0]
        Y = self._input_nodes[1]
        # Derivative with respect to the left argument.
        if(Z is X):
            return G * Y._value
        # Derivative with respect to the right argument.
        if(Z is Y):
            return G * X._value
        
    def _bprop_MSE(self, Z, G):
        """
        Backprop operation for the MSE function.
        Z: The node with respect to we want to differentiate.
        G: The gradient at the output of the operation.
        """
        T = self._input_nodes[0]
        Y = self._input_nodes[1]
    
        #N = T._value.shape[0] # Number of rows of T, needed to compute the derivative
        # Derivative with respect to the left argument.
        if(Z is T):
            return G*2*(T._value - Y._value)
        # Derivative with respect to the right argument.
        if(Z is Y):
            return -1*G*2*(T._value - Y._value)
        
        
    def _bprop_cross_entropy(self, Z, G):
        """
        Backprop operation for the cross_entropy function.
        Z: The node with respect to we want to differentiate.
        G: The gradient at the output of the operation.
        """
        T = self._input_nodes[0]
        Y = self._input_nodes[1]
    
        # Derivative with respect to the left argument.
        if(Z is T):
            return -1*G*np.log(Y._value)
        # Derivative with respect to the right argument.
        if(Z is Y):
            return -1*G*T._value/Y._value
        
        
    def _bprop_softmax(self, G):
        """
        Backprop operation for the softmax function.
        G: The gradient at the output of the operation.
        """
        Z = self._input_nodes[0]._value
        exp_of_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        num = exp_of_Z*np.sum(exp_of_Z, axis=1, keepdims=True) - exp_of_Z*exp_of_Z
        den = (np.sum(exp_of_Z, axis=1, keepdims=True))**2
        softmaxp = num/den
        return G*softmaxp
    
    
    def _bprop_softmax_xentropy(self, Z, G):
        """
        Backprop operation for the one step softmax/cross-entropy.
        It is intended to be more stable than the two step version.
        """
        T = self._input_nodes[0]._value # First argument
        X = self._input_nodes[1]._value # Second argument
        
        return G*utils.d_softmax_xentropy(T, X)
    
    def _bprop_sum(self, G):
        """
        Backprop method for sum function.
        """
        return G
    
    def _bprop_sqrt(self, G):
        """
        Backprop method for the sqrt function.
        """
        X = self._input_nodes[0]
        return G*1/(2*np.sqrt(X._value))
    
    def _bprop_sqr(self, G):
        """
        Backprop method for the sqr function.
        """
        X = self._input_nodes[0]
        return 2*G*X._value
    
    
#################################################
    

        
class Initializer:
    """
    This represents a node holding only a value, (it doesn't have an associated Operation)
    """
    def __init__(self, value):
        self.value = value
    
    def update(self, value):
        """
        Updates the value of the Initializer.
        """
        self.value = value
        
        
#################################################


class Node:
    """
    A Node holds an Operation and will be the container of the variable produced by that Operation once we ran its respective Graph.
    """
    def __init__(self, optype=None, parents=[]):
        """
        optype : The operation type that will compute the value held by the node
        parents: A list-like with the parents of this node.
        """
        
        if(get_current_graph() is None):
            # If no Graph is in the scope yet...
            set_current_graph(Graph())
            
        # Get the Graph to contain this Node
        self._graph = get_current_graph()
        # Add the Node to the Graph
        self._add_to_graph()
        
        # Set the parents
        self._parents = parents
        
        # Set the children (initially an empty list, nodes will be appended on the operation function)
        self._children = []
        
        # Build an Operation to compute the value of this Node once we run the Graph
        self._op = Operation(optype=optype, input_nodes=parents)
        
        # If this node's value is updated using the identity function, set its order to 0
        if(optype == 'identity'):
            self._order = 0
        else:
            self._order = self._graph.get_next_order()
            
    def update(self, value):
        """
        Only for initial nodes.
        Updates the value of the Node's Initializer.
        """
        parent = self._parents[0]
        if(type(parent) is Initializer):
            parent.update(value)
            
    def get_operation(self):
        """
        Returns a reference to the instance's operation.
        """
        return self._op
            
    def run(self):
        """
        Computes the value for this Node.
        """
        self._value = self._op.f()
        
    
    def _add_to_graph(self):
        """
        Adds this Node to the current Graph.
        """
        self._graph.add_node(self)
        
    def add_children(self, node):
        """
        Adds a children Node to this Node.
        """
        self._children.append(node)
        
        
    ## NODE ARITHMETIC ##
    def __add__(self, other):
        """
        Node addition.
        """
        result = Node('addition', [self,other])
        # Add result as a children of self and other
        self.add_children(result)
        other.add_children(result)
        return result
    
    def __sub__(self, other):
        """
        Node substraction.
        """
        result = Node('substraction', [self,other])
        # Add result as a children of self and other
        self.add_children(result)
        other.add_children(result)
        return result
    
    def __mul__(self, other):
        """
        Node multiplication (element wise).
        """
        result = Node('multiplication', [self,other])
        # Add result as a children of self and other
        self.add_children(result)
        other.add_children(result)
        return result
    
    def __truediv__(self, other):
        """
        Node division (element wise).
        """
        result = Node('division', [self,other])
        # Add result as a children of self and other
        self.add_children(result)
        other.add_children(result)
        return result
    
    
#################################################
        
    
# Function for the creation of initial nodes
def def_variable(value):
    """
    Creates an initial Node (order = 0), and an Initializer holding the specified value.
    Sets the Initializer as the parent of the Node, and uses as 'identity' as the operation.
    """
    initializer = Initializer(value)
    node = Node('identity', [initializer])
    return node


# Funtions to define nodes as functions of other nodes
    
def matmul(X, Y):
    """
    Returns a Node, children of X and Y, with the matmul Operation between X and Y.
    
    INPUT:
    X, Y : Parent nodes for the defined Node.
    
    OUTPUT:
    Z : The children of X and Y throught the matmul operation.
    """
    node = Node('matmul', [X,Y])
    # Append this node to the _children of X and Y
    X.add_children(node)
    Y.add_children(node)
    return node


def tanh(Z):
    """
    Returns a Node, children of Z, with the Operation tanh of Z.
    """
    node = Node('tanh', [Z])
    # Append this node to the _children of X and Y
    Z.add_children(node)
    return node

def relu(Z):
    """
    Returns a Node, children of Z, with the Operation relu of Z.
    """
    node = Node('relu', [Z])
    # Append this node to the _children of X and Y
    Z.add_children(node)
    return node


def MSE(T, Y):
    """
    Mean Squared Error between observations in T and in Y.
    """
    node = Node('mse', [T,Y])
    # Append this node to the _children of T and Y
    T.add_children(node)
    Y.add_children(node)
    return node

def cross_entropy(T, Y):
    """
    MCross entropy between empirical distributions T and Y.
    """
    node = Node('cross_entropy', [T,Y])
    # Append this node to the _children of T and Y
    T.add_children(node)
    Y.add_children(node)
    return node


def softmax_xentropy(T, Z):
    """
    MCross entropy between empirical distributions T and Y.
    """
    node = Node('softmax_xentropy', [T,Z])
    # Append this node to the _children of T and Y
    T.add_children(node)
    Z.add_children(node)
    return node


def softmax(Z):
    """
    Softmax function applied to Z.
    """
    node = Node('softmax', [Z])
    # Append this node to the _children of T and Y
    Z.add_children(node)
    return node

def mpa_sum(X):
    """
    Returns a Node, child of X, with the Operation sum of X (sums all the elements of X)
    """
    node = Node('sum', [X])
    X.add_children(node)
    return node

def sqrt(X):
    """
    Returns a Node, child of X, with the sqrt of the variable in X.
    """
    node = Node('sqrt', [X])
    X.add_children(node)
    return node

def sqr(X):
    """
    Returns a Node, child of X, with the the element-wise square of the elements in X.
    """
    node = Node('sqr', [X])
    X.add_children(node)
    return node


#################################################
    

#### Utilities to compute pruned Graphs ####
def compute_descendents(graph, initial_node, nodes): # VERIFIED
    """
    AUXILIAR FUNCTION: recursive function
    Returns a list containing the nodes that are descendents of initial_node.
    
    NOTE: To use it, nodes has to be initialized to [initial_node]
    """
    children = initial_node._children
    for child in children:
        # Only do this if child isn't already in nodes list
        if (child not in nodes):
            nodes.append(child)
            nodes = compute_descendents(graph, child, nodes)
    return nodes

def compute_ancestors(graph, final_node, nodes): # VERIFIED
    """
    AUXILIAR FUNCTION: recursive function
    Returns a list containing the nodes that are ancestors of final_node.
    
    NOTE: To use it, nodes has to be initialized to [final_node]
    """
    # Do nothing if this is an initial node
    if (final_node._order > 0):
        parents = final_node._parents
        for parent in parents:
            # Only do this if child isn't already in nodes list
            if (parent not in nodes):
                nodes.append(parent)
                nodes = compute_ancestors(graph, parent, nodes)
    return nodes

def build_pruned_graph(graph, final_node, initial_node): # VERIFIED
    """
    Returns a graph containing the nodes that are ancestors of final_node
    and descendents of initial_node.
    
    INPUT
    
    graph: The graph we want to prune.
    
    final_node: The node in the original graph for which we want to compute
    the ancestors.
    
    initial_node: The node in the original graph for which we want to compute
    the descendents.
    """
    # Get a list with the descendents of initial_node (includes initial_node)
    descendens_initial_node = compute_descendents(graph, initial_node, [initial_node])
    # Get a list with the ancestors of final_node (includes final_node)
    ancestors_final_node = compute_ancestors(graph, final_node, [final_node])
    
    # Create a Graph()
    graph = Graph()
    graph.nodes = []
    
    # If descendens_initial_node includes final_node and ancestors_final_node includes initial node...
    if ((final_node in descendens_initial_node) and (initial_node in ancestors_final_node)):
        # Include in graph the intersection between descendens_initial_node and ancestors_final_node
        nodes = set(descendens_initial_node).intersection(ancestors_final_node)
        # Cast the result (set) back to list
        graph.nodes = list(nodes)
    
    return graph

# Functions to get the inputs and the consumers of a node in a given graph
    
# Note: This function might not be necessary for the backprop algorithm (and might not be correclty formulated in the book).
def get_inputs(node, graph): # VERIFIED
    """
    Returns a list consisting of the nodes those nodes that belong to 'graph' and are parents of 'node'.
    
    INPUT:
    node: The node you want the inputs of.
    graph: The graph that restricts the available parents of 'node' you can chose from.
    """
    
    # The 'inputs' of 'node' in 'graph' are simply thouse nodes that, at the same time, belong to 'graph' and are parents of 'node'.
    node_parents = node._parents
    nodes_in_graph = graph.nodes
    node_inputs = set(node_parents).intersection(nodes_in_graph)
    # Return node_inputs as list
    return list(node_inputs)


def get_consumers(node, graph): # VERIFIED
    """
    Returns a list consisting of the nodes those nodes that belong to 'graph' and are children of 'node'.
    
    INPUT:
    node: The node you want the consumers of.
    graph: The graph that restricts the available children of 'node' you can chose from.
    """
    
    # The 'inputs' of 'node' in 'graph' are simply thouse nodes that, at the same time, belong to 'graph' and are parents of 'node'.
    node_children = node._children
    nodes_in_graph = graph.nodes
    node_consumers = set(node_children).intersection(nodes_in_graph)
    # Return node_consumers as list
    return list(node_consumers)



#################################################
    
#### BACK-PROPAGATION FUNCTION####
    
def backprop(z, target, graph):
    """
    Performs the backpropagation algorithm across the graph to compute the 
    gradient of z with respect to each node in target.
    
    INPUT:
    z : The node representing the variable we want to differentiate.
    target: a list containing the nodes with the variables with respect to we want to differtiate z.
    V : The node representing the variable with respect to we want to differtiate z.
    graph : The graph containing z and V.
    """
    # initialize grad_table with the derivative of z with respect to itself.
    grad_table = {z:1}
    
    for V in target:
        # build the pruned graph containing ancestors of z that are also descendents of V
        graph_pruned = build_pruned_graph(graph, z, V)
        # Build the gradient for V's predecesors in the computation of V (and for V itself)
        build_grad(V, graph, graph_pruned, grad_table)

    # return the gradient of z with respect to V
    return grad_table


def build_grad(V, graph, graph_pruned, grad_table):
    """
    Returns a dict containing the derivatives of z with respect to each node
    predecesor of V in the graph, and ultimately, with respect to V itself.
    The keys of the dict are references to node instances and their respective values,
    the corresponding derivatives (of z with respect to the variable represented by that node).
    
    INPUT:
    V : The variable with respect to we want to differentiate z.
    graph : The original graph, representing the set of computations that make z.
    graph_pruned : A version of graph containing only those nodes which are ancestors
    of z and predecesors of V (as well as z and V themselves).
    """
    # Check if V is alreay in grad_table. if its is, return grad_table[V]
    if (V in grad_table):
        return grad_table[V]
    
    i = 0
    # Get a list with V's consumers in graph_pruned
    consumers = get_consumers(V, graph_pruned)
    # Define an array to store the gradient of z with respect to each consumer of V in graph_pruned
    G_vec = list()
    # Loop through the consumers of V in graph_pruned
    # Get the gradient of z with respect to each consumer
    # Compute bprop, using the gradients on this node's consumers.
    for C in consumers:
        op = C.get_operation()
        D = build_grad(C, graph, graph_pruned, grad_table)
        G_vec.append(op.bprop(V, D))
        i = i + 1
    
    # Sum the gradients acorss all consumers
    G = G_vec[0]
    for i in range(1,len(G_vec)):
        G = G + G_vec[i]
    # Add the computed gradient to grad_table, return the computed gradient
    grad_table[V] = G
    return G
