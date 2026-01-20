import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
N= 1000
sigma0 = 1/5.2
gamma0 = 1/7  #recovery rate
startTime = 0
endTime = 100
timeSteps = 1000
simulationTime = np.linspace(startTime, endTime, timeSteps)
betaArray = 1.5*np.sin(simulationTime)


plt.plot(simulationTime, betaArray)

plt.xlabel("Time")
plt.ylabel("Transmission rate beta")

plt.show()
def seirStateSpaceModel(y,t,timePoints, sigma0, gamma0, betaArray):
    S, E, I =y
    beta_t = np.interp(t, timePoints, betaArray)
    dSdt = -beta_t *S*I /N
    dEdt = beta_t *S *I /N - sigma0*E
    dIdt = sigma0*E - gamma0*I
    return [dSdt, dEdt, dIdt]
   
S0, E0, I0 = 990, 7, 3
initialState = np.array([S0, E0, I0])
solutionState = odeint(seirStateSpaceModel, initialState, simulationTime, args=(simulationTime, sigma0, gamma0, betaArray))

plt.plot(simulationTime, solutionState[:,0], label='S', color = 'blue')
plt.plot(simulationTime, solutionState[:,1], label='E', color = 'red')
plt.plot(simulationTime, solutionState[:,2], label='I', color = 'green')
plt.xlabel("time")
plt.ylabel('population')
plt.show()









