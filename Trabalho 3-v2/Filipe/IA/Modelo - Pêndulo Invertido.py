import pygame
import numpy as np
import math
from scipy import linalg
import control as ct 
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import sklearn.model_selection as skm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import csv
import pandas as pd
#pip install control

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
m = 0.5
l = 0.5
L = 2*l
I = m*L**2 / 12

'''''
---Matlab----
Den = (I*(m+M) + (m*l^2)*(M+m) - (m*l)^2)*(M+m);
Deno = (I*(m+M) + (m*l^2)*(m+M) - (m*l)^2);
A = (m*l*g*(M+m))/ Deno;
B = -(m*l) / Deno;
C = (Deno + (m*l)^2)/ Den;
D = (-g*(m*l)^2)/ Deno;
Matriz_A = [[0, 1, 0, 0];[0, 0, D, 0];[0, 0, 0, 1];[0, 0, A, 0]];
Matriz_B = [[0];[C];[0];[B]];
Matriz_C = [[0, 0, 1, 0];[1, 0, 0, 0]];
Matriz_D = [[0];[0]];
[Num,Den] = ss2tf(Matriz_A, Matriz_B, Matriz_C, Matriz_D);
sys = tf(Num,Den)
rlocus(sys)
roots(Den)
Polos escolhidos foram, os dois negativos, mais 2 aleatorios. Semi-eixo negativo, estabilidade.
'''''


# SENSORES.
# sensores[0]: posição.
# sensores[1]: velocidade.
# sensores[2]: ângulo.
# sensores[3]: velocidade angular.
# SETPOINT em env.xRef.
# 



# Função de controle: Ação nula.
def funcao_controle_1(sensores):
    acao = 0
    return acao


# Função de controle.
def funcao_controle_2(sensores):
    # Controle aleatório.
    acao = 2*np.random.randn() - 1
    return acao


# Função de controle.
def funcao_controle_3(sensores):
    # Controle intuitivo.
    # Obtém valor do ângulo.
    angulo = sensores[2]
    if (angulo > 0):
        acao = +1.0
    # Se o pêndulo está caindo para a esquerda, movemos o carro para a esquerda.
    else:
        acao = -1.0
    return acao



def funcao_controle_4(sensores):        #Controle Moderno

    '''''
    Equação Lineares
    M*X2p = u - m*(X2p + l*theta2p)
    I*theta2p = m*l*g*theta - m*l*(x2p + l*theta2p)
    
    Equações Simplificadas 
    theta2p = (m*l*g*(M+m)*o)/(I*(M+m)+m*l^2*(M+m)-m^2*l^2 ) - (m*l*u)/(I*(M+m)+m*l^2*(M+m)-m^2*l^2 )
    x2p = (u*(I*(M+m)+m*l^2*(M+m)-m^2*l^2+m^2*l^2 ))/((I*(M+m)+m*l^2*(M+m)-m^2*l^2 )*(M+m) ) - (m^2*l^2*g*o))/((I*(M+m)+m*l^2*(M+m)-m^2*l^2 ) )
    '''''
    #Equações 
    Den = (I*(m+M) + (m*l**2)*(M+m) - (m*l)**2)*(M+m)
    Deno = (I*(m+M) + (m*l**2)*(m+M) - (m*l)**2)
    
    A = (m*l*g*(M+m))/ Deno
    B = -(m*l) / Deno
    C = (Deno + (m*l)**2)/ Den
    D = -(g*(m*l)**2)/ Deno
    
    #Espaço de estados -> Xp = Ax + Bu  e  Y = Cx + Du                   
    #Variaveis de estado -> x1 = x, x2 = xp, x3 = theta e x4 = thetap.      
       
    #Matriz A, é sobre a dinâmica do sistema. Matriz B, é como o sistema responde para a entrada.
    #Matrizes, a partir das variaveis de estado, para o espaço de estados  
    Matriz_A = np.array([[0, 1, 0, 0],[0, 0, D, 0],[0, 0, 0, 1],[0, 0, A, 0]])
    Matriz_B = np.array([[0],[C],[0],[B]])

    #Polos para o sistema, polo encontrado foi -4.42, +4.42, 0 e 0.
    #Muito negativo, aumenta a resposta e pouco negativo aumenta a estabilidade.
    polos = np.array([-7 + 0.23j, -4.4285, -7 - 0.23j, -15])
    
    #Calculo da matriz K, pela formula de ackerman
    k = ct.acker(Matriz_A, Matriz_B, polos)
    K = np.array(k)
    
    u = -(K[0,0]*(sensores[0]-env.xRef) + K[0,1]*sensores[1] + K[0,2]*(sensores[2]) + K[0,3]*(sensores[3]))
    acao = np.array(u)
    print()
    
    return acao

from tensorflow.keras.models import load_model
model2 = load_model(r'C:\Users\Mateus\Meu Drive\Compartilhado\eng\9_periodo\Controle inteligente\Trabalho 3-v2\Filipe\IA\testeTui10.h5')

# Cria o ambiente de simulação.
env = InvertedPendulum(0.5)
#f1 = open('entrada.csv', 'w', newline='', encoding = 'utf-8')
#w1 = csv.writer(f1)
#f2 = open('saida.csv', 'w', newline='', encoding = 'utf-8')
#w2 = csv.writer(f2)
sensor=np.array([0,0,0,0])
# Reseta o ambiente de simulação.
sensores = env.reset()
while True:
    # Renderiza o simulador.
    env.render()
    if env.finish:
        break
     
    # Calcula a ação de controle.
    sensor=np.array([sensores[0]-env.xRef, sensores[1],sensores[2],sensores[3]])
    #w1.writerow([sensores[0],sensores[1],sensores[2],sensores[3]])
    #acao = funcao_controle_4(sensores)  # É ESSA A FUNÇÃO QUE VOCÊS DEVEM PROJETAR.
    acao = model2.predict(sensor.reshape(1,4))
    #w2.writerow([acao])
    # Aplica a ação de controle.
    sensores = env.step(acao)
    #print ()
    
    

env.close()