import pygame
import numpy as np
from scipy import linalg

import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
from random import randrange 

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
        x4dot = (self.g * np.sin(x3) - np.cos(x3) * (force + self.m * self.l * x4*2 * np.sin(x3))/(self.M + self.m)) / (self.l * (4.0/3.0 - self.m * np.cos(x3)*2 / (self.M + self.m)))
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
# sensores[0]: posição. -1 1
# sensores[1]: velocidade. -20 20 
# sensores[2]: ângulo. -1 1 
# sensores[3]: velocidade angular. -10 10 
# SETPOINT em env.xRef.

Posicao = np.arange(-2, 2.01, 0.01)
Velocidade = np.arange(-75, 75.1, 0.1)
Angulo = np.arange(-1.5, 1.6, 0.1)
VelocidadeAngular = np.arange(-1000, 1000.5, 0.5)
Saida = np.arange(-1000, 1000.5, 0.5)

P = ctrl.Antecedent(Posicao, 'Posicao')
V = ctrl.Antecedent(Velocidade, 'Velocidade')
A = ctrl.Antecedent(Angulo, 'Angulo')
VA = ctrl.Antecedent(VelocidadeAngular, 'Velocidade Angular')
S = ctrl.Consequent(Saida, 'Saida')

P['NBIG'] = fuzz.trapmf(Posicao, [-2.1,-2,-0.85,-0.5])
P['NEG'] = fuzz.trimf(Posicao, [-0.75,-.575,-0.265])
P['Z'] = fuzz.trimf(Posicao, [-.375,0,.375])
P['POS'] = fuzz.trimf(Posicao, [0.265,.575,0.75])
P['PBIG'] = fuzz.trapmf(Posicao, [0.5,0.85,2,2.1])

V['NEG'] = fuzz.trimf(Velocidade, [-75,-50,0])
V['Z'] = fuzz.trimf(Velocidade, [-18,0, 18])
V['POS'] = fuzz.trimf(Velocidade, [0,50,75 ])

div=1.21 
A['NVB'] = fuzz.trapmf(Angulo,[-1.7,-1.5,-0.9,-0.6])
A['NB'] = fuzz.trimf(Angulo, [-0.825,-0.525,-0.2275])
A['N'] = fuzz.trimf(Angulo, [-0.45,-0.225,0])
A['ZO'] = fuzz.trimf(Angulo, [-0.15,0,0.15])
A['P'] = fuzz.trimf(Angulo, [0,0.225,0.45])
A['PB'] = fuzz.trimf(Angulo, [0.2275,0.525,0.825])
A['PVB'] = fuzz.trapmf(Angulo, [0.6,0.9,1.5,1.7])

#A['NVB'] = fuzz.trapmf(Angulo,[-1.7,-1.5,-1/div,-0.7/div])
#A['NB'] = fuzz.trimf(Angulo, [-1/div,-0.7/div,-0.4/div])
#A['N'] = fuzz.trimf(Angulo, [-0.6/div,-0.3/div,0/div])
#A['ZO'] = fuzz.trimf(Angulo, [-0.2/div,-0.05/div,0.2/div])
#A['P'] = fuzz.trimf(Angulo, [0/div,0.3/div,0.6/div])
#A['PB'] = fuzz.trimf(Angulo, [0.4/div,0.7/div,1/div])
#A['PVB'] = fuzz.trapmf(Angulo, [0.7/div,1/div,1.5,1.7])

VA['NB'] = fuzz.trapmf(VelocidadeAngular, [-1002,-1000,-700,-300])
VA['N'] = fuzz.trimf(VelocidadeAngular, [-600,-300,0])
VA['ZO'] = fuzz.trimf(VelocidadeAngular, [-300,0,300])
VA['P'] = fuzz.trimf(VelocidadeAngular, [0,300,600])
VA['PB'] = fuzz.trapmf(VelocidadeAngular, [300,700,1000,1002])

S['NVVB'] = fuzz.trapmf(Saida, [-1002,-1000,-800,-600])
S['NVB'] = fuzz.trimf(Saida, [-800,-600,-400])
S['NB'] = fuzz.trimf(Saida, [-600,-400,-200])
S['N'] = fuzz.trimf(Saida, [-400,-200,0])
S['Z'] = fuzz.trimf(Saida, [-200,0,200])
S['P'] = fuzz.trimf(Saida, [0,200,400])
S['PB'] = fuzz.trimf(Saida, [200,400,600])
S['PVB'] = fuzz.trimf(Saida, [400,600,800])
S['PVVB'] = fuzz.trapmf(Saida, [600,800,1000,1002])

Regra1 = ctrl.Rule(P['NBIG'] & V['NEG'], S['PVVB'])
Regra2 = ctrl.Rule(P['NEG'] & V['NEG'], S['PVB'])
Regra3 = ctrl.Rule(P['Z'] & V['NEG'], S['PB'])
Regra4 = ctrl.Rule(P['Z'] & V['Z'], S['Z'])
Regra5 = ctrl.Rule(P['Z'] & V['POS'], S['NB'])
Regra6 = ctrl.Rule(P['POS'] & V['POS'], S['NVB'])
Regra7 = ctrl.Rule(P['PBIG'] & V['POS'], S['NVVB'])

Regra8 = ctrl.Rule(A['NVB'] & VA['NB'], S['NVVB'])
Regra9 = ctrl.Rule(A['NVB'] & VA['N'], S['NVVB'])
Regra10 = ctrl.Rule(A['NVB'] & VA['ZO'], S['NVB'])
Regra11 = ctrl.Rule(A['NVB'] & VA['P'], S['NB'])
Regra12 = ctrl.Rule(A['NVB'] & VA['PB'], S['N'])
Regra13 = ctrl.Rule(A['NB'] & VA['NB'], S['NVVB'])
Regra14 = ctrl.Rule(A['NB'] & VA['N'], S['NVB'])
Regra15 = ctrl.Rule(A['NB'] & VA['ZO'], S['NB'])
Regra16 = ctrl.Rule(A['NB'] & VA['P'], S['N'])
Regra17 = ctrl.Rule(A['NB'] & VA['PB'], S['Z'])
Regra18 = ctrl.Rule(A['N'] & VA['NB'], S['NVB'])
Regra19 = ctrl.Rule(A['N'] & VA['N'], S['NB'])
Regra20 = ctrl.Rule(A['N'] & VA['ZO'], S['N'])
Regra21 = ctrl.Rule(A['N'] & VA['P'], S['Z'])
Regra22 = ctrl.Rule(A['N'] & VA['PB'], S['P'])
Regra23 = ctrl.Rule(A['ZO'] & VA['NB'], S['NB'])
Regra24 = ctrl.Rule(A['ZO'] & VA['N'], S['N'])
Regra25 = ctrl.Rule(A['ZO'] & VA['ZO'], S['Z'])
Regra26 = ctrl.Rule(A['ZO'] & VA['P'], S['P'])
Regra27 = ctrl.Rule(A['ZO'] & VA['PB'], S['PB'])
Regra28 = ctrl.Rule(A['P'] & VA['NB'], S['N'])
Regra29 = ctrl.Rule(A['P'] & VA['N'], S['Z'])
Regra30 = ctrl.Rule(A['P'] & VA['ZO'], S['P'])
Regra31 = ctrl.Rule(A['P'] & VA['P'], S['PB'])
Regra32 = ctrl.Rule(A['P'] & VA['PB'], S['PVB'])
Regra33 = ctrl.Rule(A['PB'] & VA['NB'], S['Z'])
Regra34 = ctrl.Rule(A['PB'] & VA['N'], S['P'])
Regra35 = ctrl.Rule(A['PB'] & VA['ZO'], S['PB'])
Regra36 = ctrl.Rule(A['PB'] & VA['P'], S['PVB'])
Regra37 = ctrl.Rule(A['PB'] & VA['PB'], S['PVVB'])
Regra38 = ctrl.Rule(A['PVB'] & VA['NB'], S['P'])
Regra39 = ctrl.Rule(A['PVB'] & VA['N'], S['PB'])
Regra40 = ctrl.Rule(A['PVB'] & VA['ZO'], S['PVB'])
Regra41 = ctrl.Rule(A['PVB'] & VA['P'], S['PVVB'])
Regra42 = ctrl.Rule(A['PVB'] & VA['PB'], S['PVVB'])

SistemaControle = ctrl.ControlSystem([Regra1, Regra2, Regra3, Regra4, Regra5, Regra6, Regra7, Regra8, Regra9, Regra10, 
    Regra11, Regra12, Regra13, Regra14, Regra15, Regra16, Regra17, Regra18, Regra19, Regra20, Regra21, Regra22, Regra23, Regra24, Regra25, Regra26, 
    Regra27, Regra28, Regra29, Regra30, Regra31, Regra32, Regra33, Regra34, Regra35, Regra36, Regra37, Regra38, Regra39, Regra40, Regra41, Regra42])

Controle = ctrl.ControlSystemSimulation(SistemaControle) 
# Função de controle: Ação nula.
def funcao_controle_1(sensores):
    dist=env.xRef - sensores[0] 
    Controle.input['Posicao'] = dist*5000
    Controle.input['Velocidade'] = sensores[1]*4
    Controle.input['Angulo'] = sensores[2]*4
    Controle.input['Velocidade Angular'] = sensores[3]*9
    Controle.compute()

    # dist=env.xRef - sensores[0] 
    # Controle.input['Posicao'] = dist*30
    # Controle.input['Velocidade'] = sensores[1]*8
    # Controle.input['Angulo'] = sensores[2]*8
    # Controle.input['Velocidade Angular'] = sensores[3]*3.33
    # Controle.compute()
    
    acao = Controle.output['Saida']
    #acao = Controle.defuzz(Saida, Controle., 'centroid')
    print(dist, sensores[1], sensores[2], sensores[3], acao)
    return acao, dist


# Função de controle.
def funcao_controle_2(sensores):
    # Controle aleatório.
    dist=env.xRef - sensores[0] 
    acao = 0
    print(dist, sensores[1], sensores[2], sensores[3], acao)
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


# Cria o ambiente de simulação.
#env = InvertedPendulum(randrange(-1,1))
env = InvertedPendulum(0.5)

# Reseta o ambiente de simulação.
sensores = env.reset()
acao_lista=[]
PosList = []
VelList = []
while True:
    # Renderiza o simulador.
    env.render()
    if env.finish:
        break
    
    # Calcula a ação de controle.
    acao, dist = funcao_controle_1(sensores)  # É ESSA A FUNÇÃO QUE VOCÊS DEVEM PROJETAR.
    acao_lista.append(acao)
    # PosList.append(dist)
    # VelList.append(sensores[1])
    # Aplica a ação de controle.
    sensores = env.step(acao)
    
env.close()
# plt.figure()
plt.plot(np.array(acao_lista))
# plt.figure()
# plt.plot(np.array(PosList), np.array(VelList))
plt.show()
# plt.subplot(1,3,1)
# plt.plot(VetorPosicao, VetorVelPos)
# plt.subplot(1,3,2)
# plt.plot(VetorAngulo, VetorVA)
# plt.subplot(1,3,3)
# plt.grid()
# plt.plot(X, VetorAcao),
# plt.show()
# plt.plot(X, VetorPosicao)
# plt.show()
# plt.plot(X, VetorVelPos)
# plt.show()
# plt.plot(X, VetorAngulo)
# plt.show()
# plt.plot(X, VetorVA)
# plt.show()
# plt.grid()