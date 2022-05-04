import pygame
import numpy as np
import control as c

g = 9.8
M = 1.0
m = 0.1
l = 0.5
L = 2*l
I = m*L**2 / 12

#modelagem seguindo o livro engeharia de controle moderno 5ed - Ogata

#Matriz A

a23=((m*m)*(l*l)*g)/((I+m*(l*l))*(M+m)-((m*m)*(l*l)))


a43=((m*l*g)*(M+m))/((I+m*(l*l))*(M+m)-((m*m)*(l*l)))


A =np.array([ [0, 1, 0, 0 ], [0, 0, -a23, 0], [0, 0, 0, 1], [0, 0, a43, 0]])


#A =np.array([ [0, 1, 0, 0 ], [((M+m)*g)/(M*l), 0, 0, 0], [0, 0, 0, 1], [(-m*g)/M, 0, 0, 0]])
print (A)

#Matriz B

b21=(I+m*(l*l))/((I+m*(l*l))*(M+m)-((m*m)*(l*l)))

b41=(m*l)/((I+m*(l*l))*(M+m)-((m*m)*(l*l)))

#B = np.array([[0],[(-1)/(M*l)],[0],[1/M]])

B = np.array([[0],[b21],[0],[-b41]])
print (B)

#Matriz de ganhos K (encontrados empíricamentes)
K=-1*np.array([[-1511.16, 195.7, 6576.505, 265.8395]])
#print (np.shape(K))
P2=np.array([-98.90840721+35.27678029j,-98.90840721-35.27678029j,-1.26448107,-1.55425202])
#P2=np.array([-98.90840721+35.27678029j,-98.90840721-35.27678029j,-1.26448107,-1.55425202])
#print(A-(np.dot(B,K)))
print("\n\n\n")
V,W = np.linalg.eig(A-(np.dot(B,K)))#retorna duas matrizes, uma de autovalores e outra de autovetores
print(V)#Imprime os autovalores do produto (A - B.K) que são os polos do sistema em malha fechada


G = c.acker(A, B,P2) #Utiliza a forma de Arckeman para encontrar a matriz de ganhos para comparar com os encontrados empiricamente
print (G)