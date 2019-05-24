from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]
    for i in range(N):
        # inefficient way of computing loss + grad, but pedagogical
        correct_class = y[i]
        # compute w_j^T x_i where w_j is a (D,) shaped column vector of the matrix W
        # w_j^T has shape (,D) and x_i has shape (D,) so the output is a scalar with shape ()
        
        wTx = [W[:,j].T @ X[i] for j in range(C)]
        
        # constant shift for stability
        shift = - max(wTx)
        wTx += shift
        
        # evaluate loss
        loss+= - wTx[correct_class] + np.log(np.sum(np.exp(wTx)))
        # regularize
        loss+=reg*np.sum(W**2)
        
        # compute gradient
        for k in range(C):
            dW[:,k]+= X[i] * np.exp(wTx[k])/np.sum(np.exp(wTx))
            
            # extra term if the class k is the correct class
            if k==correct_class:
                dW[:,k]+=-X[i]
            
        dW+= reg*W
        
        
    loss/=N
    dW/=N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    logits = X @ W
    N = X.shape[0]
    C = W.shape[1]
    N_rows = np.arange(N)
    # for numerical stability
    shift = - np.max(logits,1).reshape(-1,1)
    logits += shift
    correct_class_logits = logits[N_rows,y] 
    
    loss_i = - correct_class_logits + np.log(np.sum(np.exp(logits),1))
    loss += np.sum(loss_i)/N
    
    # compute gradient
    mask = np.zeros((N,C))
    mask[N_rows,y] = 1
    normalizer = np.sum(np.exp(logits),1).reshape(-1,1)
    dW += - X.T @ mask + X.T @ (np.exp(logits)/normalizer)
        
    dW += reg*W
    dW/=N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
