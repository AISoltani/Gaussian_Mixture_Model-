import numpy as np
from GMM_part1 import GMM_part1
from GMM_part2 import GMM_part2
from GMM_part3 import GMM_part3

# import test_set_hw8

d = 2 # Number of dimensions
n = 10000
# ###
# mean1 = np.matrix([[-1.], [2.]])
# covariance1 = np.matrix([
#     [1, 0.1], 
#     [0.1, 1]
# ])
# L1 = np.linalg.cholesky(covariance1)
# Y1 = np.random.normal(size=(d, n))
# X1 = L1.dot(Y1) + mean1
# ###
# mean2 = np.matrix([[3.], [2.]])
# covariance2 = np.matrix([
#     [1,  0.2], 
#     [0.2,1]
# ])
# L2 = np.linalg.cholesky(covariance2)
# Y2 = np.random.normal(size=(d, n))
# X2 = L2.dot(Y2) + mean2
# X = np.concatenate([np.array(X1),np.array(X2)],axis=1)
Y = np.load('test_set_hw8.npy') 
X = Y.T
K = 2 

Mu1,S1 = GMM_part1(X,K)
print('Part1')
print('Mu =')
print(Mu1)
print('S =')
print(S1)
Mu2,S2 = GMM_part2(X,K)
print('Part2')
print('Mu =')
print(Mu2)
print('S =')
print(S2)
Mu3,S3 = GMM_part3(X,K)
print('Part3')
print('Mu =')
print(Mu3)
print('S =')
print(S3)
