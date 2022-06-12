# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:43:00 2022

@author: Mateus
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.neural_network import MLPClassifier

nAmostras=1000
cov=0.2
cov1=np.array([[cov,0],[0,cov]])
cov2=np.array([[cov,0],[0,cov]])
mu1=np.array([-1,1])
mu2=np.array([1,-1])
X1=mvn.rvs(mean=mu1, cov=cov1, size=nAmostras//2)
X2=mvn.rvs(mean=mu2, cov=cov2, size=nAmostras//2)
X=np.concatenate([X1,X2],axis=0)
Y=np.array([0]*(nAmostras//2)+[1]*(nAmostras//2))

plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()

modelo= MLPClassifier(hidden_layer_sizes=[128,64,32],
                      activation='tanh',
                      learning_rate_init=0.01,
                      max_iter=1000
                      )

modelo.fit(X,Y)
Ypred=modelo.predict(X)

acuracia=modelo.score(X,Y)
probs=modelo.predict_proba(X)

x1=np.linspace(-4,4,1000)
x2=np.linspace(-4,4,1000)
xx, yy= np.meshgrid(x1,x2)
r1,r2=xx.flatten(),yy.flatten()
r1=r1.reshape(-1,1) 
r2=r2.reshape(-1,1)
Xgrid=np.hstack([r1,r2])
Ygrid=modelo.predict(Xgrid)
zz=Ygrid.reshape(xx.shape)

plt.contourf(xx,yy,zz)
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()
