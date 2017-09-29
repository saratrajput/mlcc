import numpy as np
import matplotlib.pyplot as plt
# for finding k eigen values
import scipy.sparse.linalg
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
################################################################################
def MixGauss(means, sigmas, n):
    """function [X, Y] = MixGauss(means, sigmas, n)

    means: (size dxp) and should be of the form [m1, ... ,mp] (each mi is
    d-dimensional)

    sigmas: (size px1) should be in the form [sigma_1;...; sigma_p]

    n: number of points per class

    X: obtained input data matrix (size 2n x d)
    Y: obtained output data vector (size 2n)

    EXAMPLE: X, Y = MixGauss(np.matrix([[0,1],[0,1]]),np.array([[0.5], [0.25]]),1000)
    generates a 2D dataset with two classes, the first one centered on (0,0)
    with standard deviation 0.5, the second one centered on (1,1)
    with standard deviation 0.25.
    Each class will contain 1000 points.

    to visualize: plt.scatter(X[:,1],X[:,2],s=25,c=Y)"""


    numRowsOfMeans = means.shape[0];
    numColsOfMeans = means.shape[1];
    
    X = []
    Y = []
    for i in range(numColsOfMeans):
        meansOfColumnI = means[:, i]
        sigmaOfI = sigmas[i]
        Xi = np.zeros((n, numRowsOfMeans))
        Yi = np.zeros((n, 1))
        
        for j in range(n):
            x = sigmaOfI * np.random.randn(numRowsOfMeans, 1) + meansOfColumnI
            Xi[j, :] = np.squeeze(np.asarray(x))
            Yi[j] = i+1
            
        X.append(Xi)
        Y.append(Yi)

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return X, Y
            
################################################################################
def PCA(X, k):
    """ V, d, X_proj = PCA(X, k)
    computes the first k eigenvectors, eigenvalues and projections of the 
    matrix (X.T.dot(X))/n where n is the number of rows in X.
    X is the dataset
    k is the number of components
     
    V is a matrix of the form [v_1, ..., v_k] where v_i is the i-th
    eigenvector
    d is the list of the first k eigenvalues
    X_proj is the projection of X on the linear space spanned by the
    eigenvectors in V"""
    
    numRowsOfX = X.shape[0]
    D, V = scipy.sparse.linalg.eigs((X.conj().transpose().dot(X))/numRowsOfX, k)
    D = D * (D > 0)
    D[::-1].sort()
    I = D.argsort()
    V = V[:, I]
    X_proj = X.dot(V)
    
    return V, D, X_proj

################################################################################

def OMatchingPursuit(X, Y, T):

    """function w, r, I = OMatchingPursuit( X, Y, T)
    Orthogonal Maching Pursuit

    X input data
    Y output labels
    T number of iterations

    w estimated coefficients
    r residuals
    I indices"""


    N, D = np.shape(X)
    
    # Initialization of residual, coefficient vector and index set I
    r = Y
    w = np.zeros((D, 1))
    I = []
    
    for i in range(T-1):
        I_tmp = list(range(D))
        
        # Select the column of X which most "explains" the residual
        a_max = -1
        
        for j in I_tmp:
            a_tmp = ((r.T.dot(X[:,j]))**2)/(X[:,j].T.dot(X[:,j]))
            
            if a_tmp > a_max:
                a_max = a_tmp
                j_max = j
                
        # Add the index to the set of indexes
        if np.sum(I == j_max) == 0:
            I.append(j_max)
            
        # Compute the M matrix
        M_I = np.zeros((D,D))
                    
        for j in I:
            M_I[j,j] = 1
                   
        A = M_I.dot(X.T).dot(X).dot(M_I)
        B = M_I.dot(X.T).dot(Y)
        
        # Update estimated coefficients
        w = np.linalg.pinv(A).dot(B)
        
        # Update the residual
        r = Y - X.dot(w)
        
    return w, r, I

################################################################################

def holdoutCVOMP(X, Y, perc, nrip, intIter):

    """l, s, Vm, Vs, Tm, Ts = holdoutCVOMP(algorithm, X, Y, kernel, perc, nrip, intRegPar, intKerPar)
    X: the training examples
    Y: the training labels
    perc: fraction of the dataset to be used for validation
    nrip: number of repetitions of the test for each couple of parameters
    intIter: range of iteration for the Orthogonal Matching Pursuit

    Output:
    it: the number of iterations of OMP that minimize the classification
    error on the validation set
    Vm, Vs: median and variance of the validation error for each couple of parameters
    Tm, Ts: median and variance of the error computed on the training set for each couple
          of parameters

    intIter = 1:50;
    Xtr, Ytr = MixGauss(np.matrix([[0,1],[0,1]]),np.array([[0.5],[0.25]]),100);
    Xtr_noise =  0.01 * np.random.randn(200, 28);
    Xtr = np.concatenate((Xtr, Xtr_noise), axis=1)
    l, s, Vm, Vs, Tm, Ts = holdoutCVOMP(Xtr, Ytr, 0.5, 5, intIter);"""

    nIter = np.size(intIter)
    
    n = X.shape[0]
    ntr = int(np.ceil(n*(1-perc)))
        
    tmn = np.zeros((nIter, nrip))
    vmn = np.zeros((nIter, nrip))
    
    for rip in range(nrip):
        I = np.random.permutation(n)
        Xtr = X[I[:ntr],:]
        Ytr = Y[I[:ntr],:]
        Xvl = X[I[ntr:],:]
        Yvl = Y[I[ntr:],:]
        
        iit = -1
        
        newIntIter = [x+1 for x in intIter]
        for it in newIntIter:
            iit = iit + 1;
            w, r, I = OMatchingPursuit(Xtr, Ytr, it)
            tmn[iit, rip] = calcErr(Xtr.dot(w),Ytr)
            vmn[iit, rip] = calcErr(Xvl.dot(w),Yvl)

            print('%-12s%-12s%-12s%-12s' % ('rip', 'Iter', 'valErr', 'trErr'))
            print('%-12i%-12i%-12f%-12f' % (rip, it, vmn[iit, rip], tmn[iit, rip]))
            
    Tm = np.median(tmn,axis=1);
    Ts = np.std(tmn,axis=1);
    Vm = np.median(vmn,axis=1);
    Vs = np.std(vmn,axis=1);
    
    # one of the min removed to make it iterable
	# nonzero returns the indices of the elements that are non-zero
    row = np.nonzero(Vm <= min(Vm));
    # added to solve last index problem
    row = row[0] 
    
    it = intIter[row[0]]
    
    return it, Vm, Vs, Tm, Ts

################################################################################
def calcErr(T, Y):
    err = np.mean(np.sign(T)!=np.sign(Y));
    return err

################################################################################


