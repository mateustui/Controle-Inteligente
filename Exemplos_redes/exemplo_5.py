import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

N=1000
X = np.linspace(-4*np.pi,4*np.pi,N)
Y=np.cos(X)
Y= Y +0.1*np.random.random(N)
X=X.reshape(N,1)


Xtrain, Xteste, Ytrain, Yteste = train_test_split(X,Y, test_size=0.3)


rede=MLPRegressor(hidden_layer_sizes=[128,32,8],
                  activation='tanh',
                  max_iter=1000)
rede.fit(Xtrain,Ytrain)

r2train=rede.score(Xtrain, Ytrain)
print("R2 Score do treinamento: ", r2train)
r2test=rede.score(Xteste, Yteste)
print("R2 Score do teste: ", r2test)

Ypred=rede.predict(Xteste)
plt.scatter(Xtrain,Ytrain)
plt.scatter(Xteste,Ypred)
plt.xlim([-4*np.pi,4*np.pi])
plt.show()  


Xn=np.linspace(-8*np.pi,8*np.pi,N*3).reshape(N*3,1)
Yn=rede.predict(Xn)
plt.scatter(Xtrain,Ytrain)
plt.plot(Xn, Yn, c='k')
plt.xlim([-8*np.pi,8*np.pi])
plt.show()