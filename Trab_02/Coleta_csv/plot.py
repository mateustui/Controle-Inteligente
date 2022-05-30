import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("teste.csv",sep=";")
display(df)
erro_posicao=df.iloc[:,0].values
velocidade=df.iloc[:,1].values
#posi_ang=df.iloc[:,2].values
#vel_ang=df.iloc[:,3].values
#acao_contro=df.iloc[:,4].values
plt.plot(erro_posicao)
#plt.plot(velocidade)