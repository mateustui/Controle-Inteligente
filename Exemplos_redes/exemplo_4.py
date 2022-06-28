import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
N=1000
X = np.linspace(0,10,N)
Y=2*X+3
Y= Y +np.random.random(N)
X=X.reshape(N,1)

'''plt.scatter(X,Y)
plt.xlim([0,10])
plt.show()
'''


Xtrain, Xteste, Ytrain, Yteste = train_test_split(X,Y, test_size=0.3)


rede=MLPRegressor(hidden_layer_sizes=[20],
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
plt.xlim([0,10])
plt.show()


Xn=np.linspace(-5,15,N).reshape(N,1)
Yn=rede.predict(Xn)
plt.scatter(Xtrain,Ytrain)
plt.plot(Xn, Yn, c='k')
plt.xlim([-5,15])
plt.show()