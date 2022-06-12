import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.neural_network import MLPClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_circles

X,Y =make_circles(n_samples=1000, noise=0.1, factor=0.5)

plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()

modelo= MLPClassifier(hidden_layer_sizes=[10,25],
                      activation='relu',
                      max_iter=1000
                      )
#modelo=LogisticRegression()

modelo.fit(X,Y)
Ypred=modelo.predict(X)

acuracia=modelo.score(X,Y)
probs=modelo.predict_proba(X)

x1=np.linspace(-2,2,1000)
x2=np.linspace(-2,2,1000)
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
