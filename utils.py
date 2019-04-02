# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 17:22:28 2018

@author: Martin Prieto Aguilera

@email: martin.canteras@gmail.com

Utils module for Simple Computational Graphs (scgraphs) library
"""
import numpy as np
import matplotlib.pyplot as plt

def oneHotEncode(t): # VERIFIED
    """
    Returns oneHotEncoded version of t  
    """
    T = np.zeros((len(t),int(np.max(t))+1))
    T[np.arange(0,len(t)),t.astype(np.int64)] = 1
    return T


def oneHotDecode(T): # VERIFIED
    """
    Returns the decoded version (t) of oneHotEncoded array T
    """
    t = np.argmax(T, axis = 1)
    return t


def relu(x):
    """
    Computes the rectified-linear operation of the components of x.
    """
    return np.fmax(0, x)

def softmax(z): # VERIFIED
    """
    The softmax activation function (stable implementation)
    Accepts a row vector, returns another row vector with the softmax of each component of the input
    """
    exp_of_Z = np.exp(z - np.max(z, axis=1, keepdims=True))
    num = exp_of_Z
    den = np.sum(exp_of_Z, axis=1, keepdims=True)
    return num/den


def logsoftmax(z):
    """
    The logsoftmax function (stable version). Accepts a row vector, and returns another row vector
    with the logsoftmax function applied to each component of the input.
    """
    b = np.max(z, axis=1, keepdims=True)
    return (z - b) - np.log(np.sum(np.exp(z - b), axis=1, keepdims=True))


def softmax_xentropy(T, Z):
    """
    Perform in one step the softmax and cross-entropy, returning only the latter.
    It is intended to be more stable from bprop's perspective.
    """
    terms = T*logsoftmax(Z)
    return -1*np.sum(terms.flatten())


def MSE(T,Y):
    """
    Computes the MSE between the (empirical) distribution of T and a model's distribution Y.
    """
    errors = np.sum((T - Y)**2, axis = 1)
    mse = np.mean(errors)
    return mse



def cross_entropy(T,Y):
    """
    Computes the cross_entropy between the (empirical) distribution of T and a model's distribution Y.
    """
    terms = T*np.log(Y)
    return np.sum(terms.flatten())

# You have to transform the 1D arrays to 2D in order to use softmax() on them:
# A = np.array([1,2])
# A.reshape(1,2)
    
def softmaxp(z): # VERIFIED
    """
    The derivative of the softmax with respect to its argument
    """
    num = np.exp(z)*np.sum(np.exp(z), axis=1, keepdims=True) - np.exp(z)*np.exp(z)
    den = (np.sum(np.exp(z), axis=1, keepdims=True))**2
    return num/den

def d_softmax_xentropy(T, Z):
    """
    The derivative of the cross_etropy w.r.t. the local field in the output layer.
    """
    N, K = np.shape(T)
    derivative = np.zeros([N,K]) # placeholder for the derivative (to return)
    # Obtain the softmax of Z
    S = softmax(Z)
    for n in range(N):
        # Get the n-th row of T (as row vector) (T_n_:) as well as the n-th row of S (S_n_:)
        T_n_a = T[n,:].reshape(1,-1)
        S_n_a = S[n,:].reshape(1,-1)
        I = np.eye(K) # Idetity matrix (K x K)
        derivative[n,:] = -1*T_n_a.dot(I - S_n_a)

    return derivative

def tanhp(z): # VERIFIED
    """
    Derivative of the hyperbolic tangent with respect to its argument
    """
    result = 1 - np.tanh(z)**2
    return result


def computeCostMSE(Y,T):
    result = np.mean(np.sum((Y - T)**2, axis=1))
    return result


def create_dummy_dataset(N=500):
    """
    Creates a dummy dataset with N observations.
    It consists of 2 non-linearly separable classes.
    """
    if (N%2 == 0):
        N1 = int(N/2)
        N2 = int(N/2)
    else:
        N1 = int(N/2)
        N2 = int(N/2) + 1
    r1 = np.random.rand(N1)*0.5
    r2 = np.random.rand(N2)*0.25 + 0.75
    
    theta1 = np.random.rand(N1)*2*np.pi
    theta2 = np.random.rand(N2)*np.pi
    
    X1_in = r1*np.cos(theta1)
    X2_in = r1*np.sin(theta1)
    X1_out = r2*np.cos(theta2)
    X2_out = r2*np.sin(theta2)
    X_in = np.vstack((X1_in,X2_in))
    X_out = np.vstack((X1_out,X2_out))
    X = np.hstack((X_in,X_out)).T
    t1 = np.zeros(N1)
    t2 = np.ones(N2)
    t = np.hstack((t1,t2))
    # Shuffle our dataset
    indices = np.arange(0,N,dtype=np.int64).tolist()
    indices_aux = list()
    while(len(indices) > 0):
        i = np.random.randint(len(indices))
        indices_aux.append(indices.pop(i))
        
    indices = indices_aux
    
    X = X[indices,:]
    t = t[indices]
    T = oneHotEncode(t)
    return X, T

def update_weights(Ws, bs, dWs, dbs, eta):
    """
    Function to update the weights given the cost derivatives with respect to weights matrices and biases vectors.
    
    INPUT:
        Ws -> Iterable containing the weights matrices.
        bs -> Iterable containing the biases vectors.
        dWs -> Iterable containing the gradients of the cost function w.r.t. the weights matrices.
        dbs -> Iterable containing the gradients of the cost function w.r.t. the biases vectors.
        eta -> learning rate
        
    OUTPUT:
        Ws_updated -> Iterable containing updated weight matrices.
        bs_updated -> Iterable containing updated biases vectors.
    """
    
    for i in range(len(Ws)):
        Ws[i] = Ws[i] - eta*dWs[i]
        bs[i] = bs[i] - eta*dbs[i]
    
    return


##### Create a function that plots the decision boundary #####
def plot_decision_boundary(X_train, X_test, t_train, t_test, graph, X_node, Y_node, T_node):
    """
    Plots the decision region on a two-feature dataset (X.shape = (N,2))
    
    INPUT:
    
    X: The dataset against which to plot the decision boundary (must be 2D)
    graph: The computational graph that calculates the signals across the MLP.
    
    t: The labels vector associated to X
    
    X_node: The node in the graph that holds the input signal
        
    Y_node: The node holding the MLP's output signal
    
    T_node: The node holding the MLP's target values used for the training stage.
    """
    # Obtain the maximum and the minimum of our dataset in each dimension
    x_min = np.min(X_train[:,0])
    y_min = np.min(X_train[:,1])
    x_max = np.max(X_train[:,0])
    y_max = np.max(X_train[:,1])

    # Create a set of points where to compute the output probability of one of the two output classes. The decision boundary
    # will be located where such probability = 0.5
    n = 200 # number of points to discretize each dimension
    x1 = np.linspace(x_min, x_max, n)
    x2 = np.linspace(y_min, y_max, n)
    X1, X2 = np.meshgrid(x1, x2)
    # Flatten the arrays so that we can construct the matrix we'll feed into our MLP
    X1_flat = X1.flatten(order='C').reshape(-1,1)
    X2_flat = X2.flatten(order='C').reshape(-1,1)
    X_in = np.hstack((X1_flat, X2_flat))

    # Run the graph on the just-built input (X_in). To do so, update the input node
    X_node.update(X_in)
    # We'll also need to update the target node so that it has the dimension of X (otherwise, we'll get an exception)
    T_node.update(np.zeros(X_in.shape))

    graph.run()

    # Obtain the output and reshape it as X1 and X2, so that we can use the contour function
    Z = Y_node._value[:,0].reshape(X1.shape)

    ##### PLOT ##### 
    plt.figure()
    # Plot the dataset
    plt.scatter(X_train[:,0], X_train[:,1], c=t_train, cmap='summer')
    plt.scatter(X_test[:,0], X_test[:,1], c=t_test, cmap='winter')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Dataset with decision boundary provided by the MLP')
    # Plot the contour
    plt.contour(X1, X2, Z, 0.5)
    plt.show()
    return
