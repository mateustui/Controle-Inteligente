import pygame
import numpy as np
from scipy import linalg
from sklearn.neural_network import MLPRegressor
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation

Dados = pd.read_csv('G:\Outros computadores\Meu modelo Computador\IFES\9 Período\Controle Inteligente\Trabalho 3\PenduloInvertidoFuzzyArtigo.csv', on_bad_lines='skip', header=None)
# Dados = pd.read_csv(r'G:\Outros computadores\Meu modelo Computador\IFES\9 Período\Controle Inteligente\Trabalho 3\sem_pertu\teste5.csv', on_bad_lines='skip', header=None)
# Dados.values
# Dados.head(5)
# print(Dados)

Entradas = Dados.iloc[:,:-1]
# Entradas.shape
Saidas = Dados.iloc[:,4]

# TENSORFLOW

Rede = Sequential()
Rede.add(Dense(64))
Rede.add(Activation('relu'))
# Rede.add(Dropout(0.5))

Rede.add(Dense(32))
Rede.add(Activation('relu'))
# Rede.add(Dropout(0.5))

Rede.add(Dense(16))
Rede.add(Activation('sigmoid'))
# Rede.add(Dropout(0.5))

Rede.add(Dense(1))

Rede.compile(loss = 'mse', #categorical_crossentropy
              optimizer = 'adam',
              metrics = ['accuracy'])
r = Rede.fit(Entradas, Saidas, epochs = 50, batch_size=10)
# TENSORFLOW


# Xtrain, Xtest, Ytrain, Ytest=train_test_split(Entradas,Saidas,test_size=0.33)
# Rede=MLPRegressor(hidden_layer_sizes=[256,128,64,32,16],
#                   activation='relu',
#                   max_iter=100000,
#                   solver="adam")
# Rede.fit(Entradas,Saidas)
# r2train=rede.score(Xtrain, Ytrain)
# print("R2 Score do treinamento: ", r2train)
# r2test=rede.score(Xtest, Ytest)
# print("R2 Score do teste: ", r2test)

# --------------------------------------------------------

# Xtrain, Xtest, Ytrain, Ytest=train_test_split(Entradas,Saidas,test_size=0.33)
# rede=MLPRegressor(hidden_layer_sizes=[128,128,64],
#                   activation='relu',
#                   max_iter=100000)
# rede.fit(Xtrain,Ytrain)
# r2train=rede.score(Xtrain, Ytrain)
# print("R2 Score do treinamento: ", r2train)
# r2test=rede.score(Xtest, Ytest)
# print("R2 Score do teste: ", r2test)

# filename = 'Rede.sav'
# pickle.dump(Rede, open(filename, 'wb'))

# --------------------------------------------------------
class InvertedPendulum():
    # Initialize environment.
    def __init__(self, xRef = 0.0, randomParameters = False, randomSensor = False, randomActuator = False):
        # System parameters.
        self.tau = 0.01
        if not randomParameters:
            self.g = 9.8
            self.M = 1.0
            self.m = 0.1
            self.l = 0.5
        else:
            self.g = 9.8 + 0.098*np.random.randn()
            self.M = 1.0 + 0.1 *np.random.randn()
            self.m = 0.1 + 0.01*np.random.randn()
            self.l = 0.5 + 0.05*np.random.randn()
            
        self.xRef = xRef

        # Drawing parameters.
        self.cartWidth = 80
        self.cartHeight = 40
        self.pendulumLength = 200
        self.baseLine = 350
        self.screenWidth = 800
        self.screenHeight = 400
        
        # Variable to see if simulation ended.
        self.finish = False
        
        # Variable to see if there is randomness in the sensors and actuators.
        self.randomSensor   = randomSensor
        self.randomActuator = randomActuator
        
        # Create a random observation.
        self.reset()

        # Create screen.
        self.screen = pygame.display.set_mode((self.screenWidth, self.screenHeight))
        pygame.display.set_caption('Inverted Pendulum')
        self.screen.fill('White')
        
        # Create a clock object.
        self.clock = pygame.time.Clock()
        pygame.display.update()

    # Close environment window.
    def close(self):
        pygame.quit()
        
    # Reset system with a new random initial position.
    def reset(self):
        self.observation = np.random.uniform(low = -0.05, high = 0.05, size = (4,))
        if self.randomSensor:
            return self.noise_sensors(self.observation.copy())
        else:
            return self.observation.copy()
    
    # Insert noise on the sensors.
    def noise_sensors(self, observation, noiseVar = 0.01):
        observation[0] = observation[0] + noiseVar*np.random.randn()
        observation[1] = observation[1] + noiseVar*np.random.randn()
        observation[2] = observation[2] + noiseVar*np.random.randn()
        observation[3] = observation[3] + noiseVar*np.random.randn()
        return observation
    
    # Insert noise on actuator.
    def noise_actuator(self, action, noiseVar = 0.01):
        action += noiseVar * np.random.randn()
        return action
    
    # Display object.
    def render(self):
        # Check for all possible types of player input.
        for event in pygame.event.get():
            # Command for closing the window.
            if (event.type == pygame.QUIT):
                pygame.quit()
                self.finish = True
                return None
            
            if (event.type == pygame.KEYDOWN):
                if (event.key == pygame.K_LEFT):
                    self.xRef -= 0.01
                    
                elif (event.key == pygame.K_RIGHT):
                    self.xRef += 0.01
                    
                elif (event.key == pygame.K_SPACE):
                    self.step(200*np.random.randn())
        
        # Apply surface over display.
        self.screen.fill('White')
        pygame.draw.line(self.screen, 'Black', (0, self.baseLine), (self.screenWidth, self.baseLine))
        
        # Get position for cart.
        xCenter = self.screenHeight + self.screenHeight * self.observation[0]
        xLeft   = xCenter - self.cartWidth//2
        # xRight  = xCenter + self.cartWidth//2
        
        # Get position for pendulum.
        pendX = xCenter +  self.pendulumLength * np.sin(self.observation[2])
        pendY = self.baseLine - self.pendulumLength * np.cos(self.observation[2])
        
        # Display objects.
        pygame.draw.line(self.screen,   'Green', (int(self.screenHeight + self.xRef * self.screenHeight), 0), (int(self.screenHeight + self.xRef * self.screenHeight), self.baseLine), width = 1)
        pygame.draw.rect(self.screen,   'Black', (xLeft, self.baseLine-self.cartHeight//2, self.cartWidth, self.cartHeight),  width = 0)
        pygame.draw.line(self.screen,   (100, 10, 10),   (xCenter, self.baseLine), (pendX, pendY), width = 6)
        pygame.draw.circle(self.screen, 'Blue',  (xCenter, self.baseLine), 10)
    
        # Draw all our elements and update everything.
        pygame.display.update()
        
        # Limit framerate.
        self.clock.tick(60)

    # Perform a step.
    def step(self, force):
        if self.randomActuator:
            force = self.noise_actuator(force)
        x1 = self.observation[0]
        x2 = self.observation[1]
        x3 = self.observation[2]
        x4 = self.observation[3]
        x4dot = (self.g * np.sin(x3) - np.cos(x3) * (force + self.m * self.l * x4**2 * np.sin(x3))/(self.M + self.m)) / (self.l * (4.0/3.0 - self.m * np.cos(x3)**2 / (self.M + self.m)))
        x2dot = (force + self.m * self.l * x4**2 * np.sin(x3))/(self.M + self.m) - self.m * self.l * x4dot * np.cos(x3) / (self.M + self.m)
        self.observation[0] = x1 + self.tau * x2
        self.observation[1] = x2 + self.tau * x2dot
        self.observation[2] = x3 + self.tau * x4
        self.observation[3] = x4 + self.tau * x4dot
        if self.randomSensor:
            return self.noise_sensors(self.observation.copy())
        else:
            return self.observation.copy()

# Parâmetros do sistema.
g = 9.8
M = 1.0
m = 0.1
l = 0.5
L = 2*l
I = m*L**2 / 12

# SENSORES.
# sensores[0]: posição.
# sensores[1]: velocidade.
# sensores[2]: ângulo.
# sensores[3]: velocidade angular.
# SETPOINT em env.xRef.


    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(sensores[0], sensores[1], sensores[2], sensores[3])
Sensores_sim=np.array([0,0,0,0])
# Função de controle: Ação nula.
def funcao_controle_1(sensores):
    
    # Sensores_sim[0]=env.xRef-sensores[0]
    # Sensores_sim[1]=sensores[1]
    # Sensores_sim[2]=sensores[2]
    # Sensores_sim[3]=sensores[3]
    sensores_sim = np.array([sensores[0]-env.xRef, sensores[1],sensores[2],sensores[3]])
    acao=Rede.predict(sensores_sim.reshape(1,4))
    print(acao)
    return acao


# Cria o ambiente de simulação.
env = InvertedPendulum(0.50)

# Reseta o ambiente de simulação.
sensores = env.reset()

while True:
    # Renderiza o simulador.
    env.render()
    if env.finish:
        break
    
    # Calcula a ação de controle.
    acao = funcao_controle_1(sensores)  # É ESSA A FUNÇÃO QUE VOCÊS DEVEM PROJETAR.
    
    # Aplica a ação de controle.
    sensores = env.step(acao)
    
env.close()