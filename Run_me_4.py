import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from GMM import GMM 
import math


train_images = 'train-images.idx3-ubyte'
train_labels = 'train-labels.idx1-ubyte'
test_images = 't10k-images.idx3-ubyte'
test_labels = 't10k-labels.idx1-ubyte'

train_images = idx2numpy.convert_from_file(train_images)
train_labels = idx2numpy.convert_from_file(train_labels)
test_images = idx2numpy.convert_from_file(test_images)
test_labels = idx2numpy.convert_from_file(test_labels)

print("train images size: ",train_images.shape)
print("train labels size: ",train_labels.shape)
print("test images size: ",test_images.shape)
print("test labels size: ",test_labels.shape)

# for i in range(11):
#     print(train_labels[i])

# train_images = train_images/255
# train_images = train_images.reshape(train_images.shape[0],784)

idx = np.argsort(train_labels)
x_train_sorted = train_images[idx]
y_train_sorted = train_labels[idx]
## 0
x_train_zeros = train_images[train_labels == 0]
x_train_zeros = x_train_zeros/255 
x_train_zeros = x_train_zeros.reshape(x_train_zeros.shape[0],784)
L0 = np.linalg.cholesky(1e-3 * np.eye(784))
Y0 = np.random.normal(size=(784, x_train_zeros.shape[0]))
X0 = L0.dot(Y0)
x_train_zeros = x_train_zeros + X0.T

## 1
x_train_ones = train_images[train_labels == 1]
x_train_ones = x_train_ones/255
x_train_ones = x_train_ones.reshape(x_train_ones.shape[0],784)
L1 = np.linalg.cholesky(1e-3 * np.eye(784))
Y1 = np.random.normal(size=(784, x_train_ones.shape[0]))
X1 = L1.dot(Y1)
x_train_ones = x_train_ones + X1.T
## 2
x_train_twos = train_images[train_labels == 2]
x_train_twos = x_train_twos/255
x_train_twos = x_train_twos.reshape(x_train_twos.shape[0],784)
L2 = np.linalg.cholesky(1e-3 * np.eye(784))
Y2 = np.random.normal(size=(784, x_train_twos.shape[0]))
X2 = L2.dot(Y2)
x_train_twos = x_train_twos + X2.T
## 3
x_train_threes = train_images[train_labels == 3]
x_train_threes = x_train_threes/255
x_train_threes = x_train_threes.reshape(x_train_threes.shape[0],784)
L3 = np.linalg.cholesky(1e-3 * np.eye(784))
Y3 = np.random.normal(size=(784, x_train_threes.shape[0]))
X3 = L3.dot(Y3)
x_train_threes = x_train_threes + X3.T
## 4
x_train_fours = train_images[train_labels == 4]
x_train_fours = x_train_fours/255
x_train_fours = x_train_fours.reshape(x_train_fours.shape[0],784)
L4 = np.linalg.cholesky(1e-3 * np.eye(784))
Y4 = np.random.normal(size=(784, x_train_fours.shape[0]))
X4 = L4.dot(Y4)
x_train_fours = x_train_fours + X4.T
## 5
x_train_fives = train_images[train_labels == 5]
x_train_fives = x_train_fives/255
x_train_fives = x_train_fives.reshape(x_train_fives.shape[0],784)
L5 = np.linalg.cholesky(1e-3 * np.eye(784))
Y5 = np.random.normal(size=(784, x_train_fives.shape[0]))
X5 = L5.dot(Y5)
x_train_fives = x_train_fives + X5.T
## 6
x_train_sixs = train_images[train_labels == 6]
x_train_sixs = x_train_sixs/255
x_train_sixs = x_train_sixs.reshape(x_train_sixs.shape[0],784)
L6 = np.linalg.cholesky(1e-3 * np.eye(784))
Y6 = np.random.normal(size=(784, x_train_sixs.shape[0]))
X6 = L6.dot(Y6)
x_train_sixs = x_train_sixs + X6.T
## 7
x_train_sevens = train_images[train_labels == 7]
x_train_sevens = x_train_sevens/255
x_train_sevens = x_train_sevens.reshape(x_train_sevens.shape[0],784)
L7 = np.linalg.cholesky(1e-3 * np.eye(784))
Y7 = np.random.normal(size=(784, x_train_sevens.shape[0]))
X7 = L7.dot(Y7)
x_train_sevens = x_train_sevens + X7.T
## 8
x_train_eights = train_images[train_labels == 8]
x_train_eights = x_train_eights/255
x_train_eights = x_train_eights.reshape(x_train_eights.shape[0],784)
L8 = np.linalg.cholesky(1e-3 * np.eye(784))
Y8 = np.random.normal(size=(784, x_train_eights.shape[0]))
X8 = L8.dot(Y8)
x_train_eights = x_train_eights + X8.T
## 9
x_train_nines = train_images[train_labels == 9]
x_train_nines = x_train_nines/255
x_train_nines = x_train_nines.reshape(x_train_nines.shape[0],784)
L9 = np.linalg.cholesky(1e-3 * np.eye(784))
Y9 = np.random.normal(size=(784, x_train_nines.shape[0]))
X9 = L9.dot(Y9)
x_train_nines = x_train_nines + X9.T

###############################
test_images = test_images/255
test_images = test_images.reshape(test_images.shape[0],784)
train_images = train_images/255
train_images = train_images.reshape(train_images.shape[0],784)
###############################
K = 5
S0,Mu0,Pi0,l0 = GMM (x_train_zeros.T,K)
while (math.isnan(l0)):
    S0,Mu0,Pi0,l0 = GMM (x_train_zeros.T,K)

S1,Mu1,Pi1,l1 = GMM (x_train_ones.T,K)
while (math.isnan(l1)):
    S1,Mu1,Pi1,l1 = GMM (x_train_ones.T,K)
S2,Mu2,Pi2,l2 = GMM (x_train_twos.T,K)
while (math.isnan(l2)):
    S2,Mu2,Pi2,l2 = GMM (x_train_twos.T,K)
S3,Mu3,Pi3,l3 = GMM (x_train_threes.T,K)
while (math.isnan(l3)):
    S3,Mu3,Pi3,l3 = GMM (x_train_threes.T,K)
S4,Mu4,Pi4,l4 = GMM (x_train_fours.T,K)
while (math.isnan(l4)):
    S4,Mu4,Pi4,l4 = GMM (x_train_fours.T,K)
S5,Mu5,Pi5,l5 = GMM (x_train_fives.T,K)
while (math.isnan(l5)):
    S5,Mu5,Pi5,l5 = GMM (x_train_fives.T,K)
S6,Mu6,Pi6,l6 = GMM (x_train_sixs.T,K)
while (math.isnan(l6)):
    S6,Mu6,Pi6,l6 = GMM (x_train_sixs.T,K)
S7,Mu7,Pi7,l7 = GMM (x_train_sevens.T,K)
while (math.isnan(l7)):
    S7,Mu7,Pi7,l7 = GMM (x_train_sevens.T,K)
S8,Mu8,Pi8,l8 = GMM (x_train_eights.T,K)
while (math.isnan(l8)):
    S8,Mu8,Pi8,l8 = GMM (x_train_eights.T,K)

S9,Mu9,Pi9,l9 = GMM (x_train_nines.T,K)
while (math.isnan(l9)):
    S9,Mu9,Pi9,l9 = GMM (x_train_nines.T,K)


G_D0 = 0
G_D1 = 0
G_D2 = 0
G_D3 = 0
G_D4 = 0
G_D5 = 0
G_D6 = 0
G_D7 = 0
G_D8 = 0
G_D9 = 0


for k in range(K):
    G_D0 = G_D0 + Pi0[k]*((-1/2)*(np.sum(np.log(S0[k])))+(-1/2)*np.sum(((test_images-Mu0[k].T)*(1/S0[k])*(test_images-Mu0[k].T)),axis=1))
    G_D1 = G_D1 + Pi1[k]*((-1/2)*(np.sum(np.log(S1[k])))+(-1/2)*np.sum(((test_images-Mu1[k].T)*(1/S1[k])*(test_images-Mu1[k].T)),axis=1))
    G_D2 = G_D2 + Pi2[k]*((-1/2)*(np.sum(np.log(S2[k])))+(-1/2)*np.sum(((test_images-Mu2[k].T)*(1/S2[k])*(test_images-Mu2[k].T)),axis=1))
    G_D3 = G_D3 + Pi3[k]*((-1/2)*(np.sum(np.log(S3[k])))+(-1/2)*np.sum(((test_images-Mu3[k].T)*(1/S3[k])*(test_images-Mu3[k].T)),axis=1))
    G_D4 = G_D4 + Pi4[k]*((-1/2)*(np.sum(np.log(S4[k])))+(-1/2)*np.sum(((test_images-Mu4[k].T)*(1/S4[k])*(test_images-Mu4[k].T)),axis=1))
    G_D5 = G_D5 + Pi5[k]*((-1/2)*(np.sum(np.log(S5[k])))+(-1/2)*np.sum(((test_images-Mu5[k].T)*(1/S5[k])*(test_images-Mu5[k].T)),axis=1))
    G_D6 = G_D6 + Pi6[k]*((-1/2)*(np.sum(np.log(S6[k])))+(-1/2)*np.sum(((test_images-Mu6[k].T)*(1/S6[k])*(test_images-Mu6[k].T)),axis=1))
    G_D7 = G_D7 + Pi7[k]*((-1/2)*(np.sum(np.log(S7[k])))+(-1/2)*np.sum(((test_images-Mu7[k].T)*(1/S7[k])*(test_images-Mu7[k].T)),axis=1))
    G_D8 = G_D8 + Pi8[k]*((-1/2)*(np.sum(np.log(S8[k])))+(-1/2)*np.sum(((test_images-Mu8[k].T)*(1/S8[k])*(test_images-Mu8[k].T)),axis=1))
    G_D9 = G_D9 + Pi9[k]*((-1/2)*(np.sum(np.log(S9[k])))+(-1/2)*np.sum(((test_images-Mu9[k].T)*(1/S9[k])*(test_images-Mu9[k].T)),axis=1))


p0 =  x_train_zeros.shape[0]/60000
p1 =  x_train_ones.shape[0]/60000
p2 =  x_train_twos.shape[0]/60000
p3 =  x_train_threes.shape[0]/60000
p4 =  x_train_fours.shape[0]/60000
p5 =  x_train_fives.shape[0]/60000
p6 =  x_train_sixs.shape[0]/60000
p7 =  x_train_sevens.shape[0]/60000
p8 =  x_train_eights.shape[0]/60000
p9 =  x_train_nines.shape[0]/60000

bbb = np.array([p0 * G_D0,p1 * G_D1,p2 * G_D2,p3 * G_D3,p4 * G_D4,p5 * G_D5,p6 * G_D6,p7 * G_D7,p8 * G_D8,p9 * G_D9])
np.max(bbb,axis=0)
error=((np.argmax(bbb,axis=0) != test_labels).sum())/10000 * 100
print(error)
