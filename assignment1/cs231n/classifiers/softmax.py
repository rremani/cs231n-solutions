import numpy as np
from random import shuffle

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


  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    #correct_class_score = scores[y[i]]
    #count = 0

    scores -= np.max(scores) #to account for numerical instability
    sum_i = 0
    for k in scores:
      sum_i += np.exp(k)

    loss += - scores[y[i]] + np.log(sum_i)
    #p = np.exp(scores) / np.sum(np.exp(scores))
    for j in xrange(num_classes):
      p = np.exp(scores[j])/sum_i
      if j==y[i]:
        dW[:,j] += X[i] * p - X[i]
      else:
        dW[:,j] += X[i] * p
      #dW[:,j] += (p-(j == y[i])) * X[i,:]



  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  
  # Add regularization to the loss and the gradient
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores) # to normalizing to overcome numerical instability
  scores_correct = scores[range(num_train),y]
  loss = np.mean(-scores_correct + np.log(np.sum(np.exp(scores),axis=1)))
  p = np.exp(scores)/(np.sum(np.exp(scores),axis=1)[:,np.newaxis])
  
  index = np.zeros(p.shape)
  index[range(num_train),y]=1

  dW = (np.dot((p-index).T,X)).T
  dW /= num_train

  # Regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  return loss, dW

  
  
  

