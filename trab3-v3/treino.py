import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as skm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


Dados = pd.read_csv(r'C:\Users\Mateus\Meu Drive\Compartilhado\eng\9_periodo\Controle inteligente\trab3-v3\teste5.csv', on_bad_lines='skip', header=None)
Entrada = Dados.iloc[:,:-1]
Saida = Dados.iloc[:,4]
Xtrain, Xtest, Ytrain, Ytest = skm.train_test_split(Entrada, Saida, test_size = 0.33)


modelo = Sequential()
modelo.add(Dense(128))
modelo.add(Activation('relu'))
modelo.add(Dense(128))
modelo.add(Activation('relu'))
modelo.add(Dense(16))
modelo.add(Activation('sigmoid'))
modelo.add(Dense(1))

modelo.compile(optimizer = 'adam', loss = 'mse')

hist = modelo.fit(Xtrain, Ytrain,
                  epochs = 500,
                  batch_size = 100,
                  
                  validation_data = (Xtest, Ytest))

plt.plot(hist.history['loss']) 
plt.plot(hist.history['val_loss']) 
plt.grid()
plt.xlabel('Épocas')
plt.ylabel('Erro médio quadrático')
plt.xlim([0, 500])
plt.show()

modelo.save(r'testeTui10.h5')