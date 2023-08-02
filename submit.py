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

def mse(w, x, y, n):
    e = 0

    for i in range(n):
        e = e + np.power((np.dot(w, x[i]) - y[i]), 2)/(2*n)

    return e

def delta_w(w, x, y, n):
    e = 0
    del_w = np.zeros(len(w))
    
    for j in range(len(w)):
	    for i in range(n):
		    del_w[j] = del_w[j] + (np.dot(w, x[i]) - y[i])*x[i][j]/n

    return del_w


################################
# Non Editable Region Starting #
################################
import numpy as np

def my_fit(X_trn, y_trn):
    ################################
    #  Non-Editable Region Ending  #
    ################################

    # Use this method to train your model using training CRPs
    # Your method should return a 2048-dimensional vector that is 512-sparse
    # No bias term allowed -- return just a single 2048-dim vector as output
    # If the vector you return is not 512-sparse, it will be sparsified using hard-thresholding

    D = 2048
    S = 512
    n = len(y_trn)
    w0 = np.random.rand(D)
    
    t=20
    
    while t :
        print(mse(w0, X_trn, y_trn, n))
        z = w0 - 0.5*delta_w(w0,X_trn,y_trn,n)
        w0 = HT (z,S)
        t = t-1
         
    # print(X_trn[0],y_trn[0])

    return w0  # Return the trained model


X_trn = np.loadtxt("train_challenges.dat")
y_trn = np.loadtxt("train_responses.dat")

my_fit(X_trn, y_trn)
