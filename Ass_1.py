import numpy as np

# You are not allowed to use any ML libraries e.g. sklearn, scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SKLEARN, SCIPY, KERAS,TENSORFLOW ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHOD my_fit BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

def HT( v, k ):
    t = np.zeros_like( v )
    if k < 1:
        return t
    else:
        ind = np.argsort( abs( v ) )[ -k: ]
        t[ ind ] = v[ ind ]
        return t
    
def mse(w, x, y):
    n = x.shape[0]
    error = np.dot(x, w) - y
    squared_error = np.sum(np.power(error, 2))
    mse = squared_error / (2 * n)
    return mse
  

################################
# Non Editable Region Starting #
################################
def my_fit(X_trn, y_trn, alpha):
################################
#  Non-Editable Region Ending  #
################################

# Use this method to train your model using training CRPs
# Your method should return a 2048-dimensional vector that is 512-sparse
# No bias term allowed -- return just a single 2048-dim vector as output
# If the vector you return is not 512-sparse, it will be sparsified using hard-thresholding

    D = X_trn.shape[1]
    n = X_trn.shape[0]
    S = 512
    # w0 = np.zeros(D)
    w0=np.linalg.lstsq( X_trn, y_trn, rcond = None )[0]
    
    t=1000
    msep=0
    msec=1e7
    
    while t and abs(msec-msep)/msec*100 > 0.01 :
        msep=msec
        msec=mse(w0, X_trn, y_trn)
        
        diff = np.dot(X_trn, w0) - y_trn
        gradient = np.dot(diff, X_trn) / n
        
        z = w0 - alpha*gradient
        
        w0=np.maximum(w0,0)
        w0 = HT (z,S)
        t = t-1

    print(msec) 

    return w0  # Return the trained model
