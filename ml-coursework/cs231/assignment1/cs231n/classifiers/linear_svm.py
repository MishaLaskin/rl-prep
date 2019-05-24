from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin 
                dW[:,j] += X[i]
                dW[:,y[i]] -= X[i] # results in dW[:,y[i]] -= counts*X[i] at the end
        
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # we kept most of the gradient calculations in the loop above
    # using gradients from here http://cs231n.github.io/optimization-1/#gd
    
    for j in range(num_classes):
        dW[:,j]+=reg*W[:,j] # add the gradient due to regularization
    dW/=num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # one hot encode the classes so that y_one_hot has shape (N_data,N_classes)
    # zero out indices of all classes except for the correct one
    # STOPPED HERE THIS LOSS IS NOT COMPLETE)
    num_classes = W.shape[1]
    N = X.shape[0]
    N_rows = np.arange(N)
    #y_one_hot = np.eye(num_classes)[y]
    # scores per class for each data point - shape (N,c)
    scores = X @ W
    # only scores for the correct class - shape (N,1) - 1 needed for broadcasting
    y_scores = scores[N_rows,y].reshape(-1,1)
    Delta = 1.0
    # compute the margin - shape (N,c)
    margin = np.maximum(0,scores - y_scores + Delta)
    # margin is a sum over all j!=y_i
    # set all j=y_i to zero
    margin[N_rows,y] = 0
    # loss L = \sum_i L_i for i in N 
    # and L_i = sum_{j!=y_i} hinge_loss
    loss = np.sum(margin)/N + reg*np.sum(W**2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW+= 2*reg*W
    # X_indicator_func is the matrix form of the indicator function
    # in the gradient loss 
    # it has shape (N,c)
    
    X_indicator_func = np.zeros_like(margin)
    X_indicator_func[margin>0]=1
    # count the number of correct class y_i nonzero margin
    y_counts = np.sum(X_indicator_func,1)
    X_indicator_func[N_rows,y]-=y_counts
    # to get final gradient, multiply indicator func by x_i for each x_i
    dW+=X.T @ X_indicator_func
    dW/=N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
