# Import Libraries
import numpy as np
def GMM_part3 (x,K):
    
    l = 10**20
    while np.abs(l)>10**15:
        n = x.shape[1]
        d = x.shape[0]
        TOL = 10**-15
        Pi = np.random.normal(size = K)
        Pi = np.exp(Pi)
        Pi /= Pi.sum()
        Pi = np.reshape(Pi,-1)
        
        Mu = []
        S = []
        SS = []
        # diag_S = []
        inv_S = []
        # diag_inv_S = []
        Loss = []
        Mu = np.random.normal(size=(d, K))
    
        for k in range(K):

            s = np.abs(np.random.normal(size=(d,1)))
            inv_s = 1/s
            S.append(s)
            SS.append(s*np.eye(d))
            inv_S.append(inv_s)
        for j  in range(d):
            X = x[j,:]
            Max_iter = 500
            r = np.zeros([n,K])
            for iteration in range(Max_iter):
                for k in range(K):
                    r[:,k] = Pi[k]*((S[k][j])**(-1/2))*(np.exp((-1/2)*(X-Mu[j,k]) * (X-Mu[j,k]) / S[k][j]))
                    # print(r.shape,r)
                
                
                R1 = np.reshape(np.sum(r,axis=1),(n,1)) #ri.
                r = r/R1
                l = -np.log(R1).sum()
                if np.abs(l)>10**15:
                    break
                Loss.append(l)
                if iteration > 1 and np.abs(np.abs(Loss[iteration])-np.abs(Loss[iteration-1]))<=TOL*np.abs(Loss[iteration]):
                    break
                R0 = np.reshape(np.sum(r,axis=0),(K)) #r.k
                Pi = R0/n
                for k in range(K):
                    Mu[j,k] = np.dot(r[:,k],X)/R0[k]
                    S[k][j,0] = np.sum(r[:,k]*X**2)/R0[k]-(Mu[j,k]**2)
                    # print(S[k])
                    SS[k] = S[k]*np.eye(d)
    
    # print('Mu=',Mu.T)
    # print('S=',SS)
    return Mu.T,SS
    
