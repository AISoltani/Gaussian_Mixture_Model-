import numpy as np

def GMM_part2 (X,K):

    l = 10**20
    while np.abs(l)>10**15:
        n = X.shape[1]
        d = X.shape[0]
        TOL = 10**-10
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
        for k in range(K):
            mu = np.random.normal(size=(d, 1))
            Mu.append(mu)
            s = np.ones(1)
            inv_s = 1/s
            S.append(s)
            SS.append(s*np.eye(d))
        Max_iter = 500
        r = np.zeros([n,K])
        for iteration in range(Max_iter):
            for k in range(K):
                r[:,k] = Pi[k]*((S[k])**(-d/2))*(np.exp((-1/2)*np.sum((X-Mu[k]) * (X-Mu[k]) / S[k],axis=0)))
                
            
            R1 = np.reshape(np.sum(r,axis=1),(n,1)) #ri.
            r = r/R1
            # r = r +10**-8
            l = -np.log(R1).sum()
            if np.abs(l)>10**15:
                break
            Loss.append(l)
            if iteration > 1 and np.abs(Loss[iteration]-Loss[iteration-1])<=TOL*np.abs(Loss[iteration]):
                break
            R0 = np.reshape(np.sum(r,axis=0),(K)) #r.k
            Pi = R0/n
            for k in range(K):
                Mu[k] = np.reshape(np.sum(r[:,k]*X,axis=1)/R0[k],(d,1))
                S[k] = (np.sum(np.sum(r[:,k]*X**2,axis=1)/R0[k]-(Mu[k].T**2)))/d
                SS[k] = S[k]*np.eye(d)
    
    
    # print('Mu=',Mu)
    # print('S=',SS)
    return Mu,SS 