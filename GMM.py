
import numpy as np
import math

 
def GMM(X,K):
    l = 10**20
    while np.abs(l)>10**15:
        n = X.shape[1]
        d = X.shape[0]
        TOL = 10**-8       
        Pi = np.random.normal(size = K)
        Pi = np.exp(Pi)
        Pi /= Pi.sum()
        Pi = np.reshape(Pi,-1)
        Mu = []
        S = []
        SS = []
        inv_S = []
        Loss = []
        for k in range(K):
            mu = 0* np.abs(np.random.normal(size=(d, 1)))
            Mu.append(mu)

            
            s = np.ones((d,1))
            inv_s = 1/s
            S.append(s)
            SS.append(s*np.eye(d))
            inv_S.append(inv_s)
        Max_iter = 500
        r = np.zeros([n,K])
        for iteration in range(Max_iter):
            for k in range(K):
                Inv_s = np.reshape(inv_S[k],(d,1))
                r[:,k] = np.log(Pi[k])+(-1/2)*(np.sum(np.log(S[k])))+((-1/2)*(np.sum((X-Mu[k]) * Inv_s * (X-Mu[k]),axis=0)))  #Inv_s is a vector

            R1 = np.reshape(np.sum(r,axis=1),(n,1)) #ri.
            r = r/R1
            l = -R1.sum()
            if (math.isnan(l)):
                break   
            Loss.append(l)
            if iteration > 1 and np.abs(np.abs(Loss[iteration])-np.abs(Loss[iteration-1]))<=TOL*np.abs(Loss[iteration]):
                break
            R0 = np.reshape(np.sum(r,axis=0),(K)) #r.k
            Pi = R0/n
            for k in range(K):
                Mu[k] = np.reshape(np.sum(r[:,k]*X,axis=1)/R0[k],(d,1))
                S[k] = np.sum(r[:,k]*X**2,axis=1)/R0[k]-(Mu[k].T**2)
                inv_S[k] = (S[k])**(-1)
                SS[k] = S[k]*np.eye(d)
    return  S, Mu,Pi,l