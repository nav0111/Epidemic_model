import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

N=1000
S0 = 990
E0 =5
I0 =5

beta0 =0.9
sigma0 = 1/5.2
gamma0 = 1/7

T = np.linspace(1,100,100)
y0 =np.array([S0, E0, I0])

def seirModel(y,t, beta0, sigma0, gamma0,N):
    S, E, I =y
    dSdt = -beta0 *S *I/N
    dEdt = beta0 *S *I/N - sigma0*E
    dIdt = sigma0*E - gamma0*I
    return [dSdt, dEdt, dIdt]

solutionState = odeint(seirModel, y0, T, args =(beta0, sigma0, gamma0, N))

plt.plot(T, solutionState[:,0], label= 'Susceptible', color ='blue')
plt.plot(T, solutionState[:,1], label = 'Exposed', color='red')
plt.plot(T, solutionState[:,2], label ='Infected', color = 'green')
plt.xlabel("Days")
plt.ylabel("population")
plt.show()