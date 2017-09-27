import numpy as np
import matplotlib.pyplot as plt
#from scipy import *
import scipy as sp
#from numpy import *
# for finding k eigen values
import scipy.sparse.linalg
################################################################################
def MixGauss(means, sigmas, n):
    numRowsOfMeans = means.shape[0];
    numColsOfMeans = means.shape[1];
    
#     X = zeros((200,2))
#     Y = zeros((200,1))
    
#     X = zeros((200,2))
#     Y = zeros((200,1))
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
            
#         tempXi = Xi
#         tempYi = Yi
            
        X.append(Xi)
        Y.append(Yi)
#         X = column_stack(Xi)
#         X = concatenate((Xi, Xi), axis=0)
#         Y = concatenate((Yi, Yi), axis=0)
#         X[0:100,:] = tempXi
#         X[100:200,:] = Xi
#         Y[0:100,:] = tempYi
#         Y[100:200,:] = Yi
    return X, Y
            
################################################################################
def PCA(X, k):
# [V, d, X_proj] = PCA(X, k)
# computes the first k eigenvectors, eigenvalues and projections of the 
# matrix X'*X/n where n is the number of rows in X.
# 
# X is the dataset
# k is the number of components
# 
# V is a matrix of the form [v_1, ..., v_k] where v_i is the i-th
# eigenvector
# d is the list of the first k eigenvalues
# X_proj is the projection of X on the linear space spanned by the
# eigenvectors in V
    
    numRowsOfX = X.shape[0]
    D, V = scipy.sparse.linalg.eigs((X.conj().transpose().dot(X))/numRowsOfX, k)
#     diagonal = diag(D)
    D = D * (D > 0)
#     D = sorted(D, reverse = True)
    D[::-1].sort()
    I = D.argsort()
    V = V[:, I]
    X_proj = X.dot(V)
    
    return V, D, X_proj

################################################################################

def OMatchingPursuit(X, Y, T):

    N, D = np.shape(X)
    
    # Initialization of residual, coefficient vector and index set I
    r = Y
    w = np.zeros((D, 1))
    I = []
    
    for i in range(T-1):
        I_tmp = list(range(D))
        
        ### Select the column of X which most "explains" the residual
        a_max = -1
        
        for j in I_tmp:
#             a_tmp = ((residual.T * X[:,1])**2)/(X[:,1].T * X[:,1])
            a_tmp = ((r.T.dot(X[:,j]))**2)/(X[:,j].T.dot(X[:,j]))
#             a_tmp = dot(residual.T, inputData[:])
            
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
#         A = dot((dot(dot(M_I, X.T), X)), M_I)
        B = M_I.dot(X.T).dot(Y)
#         B = dot((dot(M_I, X.T)), Y)
        
        # Update estimated coefficients
        w = np.linalg.pinv(A).dot(B)
        
        # Update the residual
        r = Y - X.dot(w)
#         residual = outputLabel - dot(inputData, estimatedCoeff)
        
    return w, r, I

################################################################################

def holdoutCVOMP(X, Y, perc, nrip, intIter):
    nIter = size(intIter)
    
    n = X.shape[0]
    ntr = int(ceil(n*(1-perc)))
        
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
            
#             print(('rip\\tIter\\tvalErr\\ttrErr\\n%d\\t%d\\t%f\\t%f\\n'), rip, it, vmn(iit, rip), tmn(iit, rip))
            
    Tm = median(tmn,axis=1);
    Ts = std(tmn,axis=1);
    Vm = median(vmn,axis=1);
    Vs = std(vmn,axis=1);
    
    # one of the min removed to make it iterable
    row = nonzero(Vm <= min(Vm));
    # added to solve last index problem
    row = row[0] 
    
    it = intIter[row[0]]
    
    return it, Vm, Vs, Tm, Ts

################################################################################
def calcErr(T, Y):
    err = np.mean(np.sign(T)!=np.sign(Y));
    return err

################################################################################


