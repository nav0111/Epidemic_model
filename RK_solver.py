import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import torch



#plt.plot(simulationTime, betaArray)
#plt.xlabel("Time")
#plt.ylabel("Transmission rate beta")
#plt.show()

def sigmoid(x, a, b):
    x = np.asarray(x)
    return 1 / (1 + np.exp(-a*(x-b)))

def create_beta_array(time_domain):
    a = 0.1 
    b = 0.5
    beta_min  = 0.1 
    beta_max = 1.1
    beta_array = beta_min + (beta_max - beta_min) * sigmoid(time_domain, a, b)
    return beta_array

def seirStateSpaceModel(y,t,timePoints, sigma0, gamma0, betaArray, N):
    S, E, I =y
    beta_t = np.interp(t, timePoints, betaArray)
    print(beta_t)
    #beta_t = betaArray[t]
    dSdt = -beta_t *S*I /N
    dEdt = beta_t *S *I /N - sigma0*E
    dIdt = sigma0*E - gamma0*I
    return [dSdt, dEdt, dIdt]

def run_seir_model(): 
    # define parameters
    N= 1000
    sigma0 = 1/5.2
    gamma0 = 1/7  #recovery rate
    startTime = 0
    endTime = 100
    timeSteps = 1000
    simulationTime = np.linspace(startTime, endTime, timeSteps)

    beta_array = create_beta_array(simulationTime)
    
    # define initial conditions
    S0, E0, I0 = 990, 7, 3 
    
    # run simulation
    initialState = np.array([S0, E0, I0])
    solutionState = odeint(seirStateSpaceModel, initialState, simulationTime, args=(simulationTime, sigma0, gamma0, beta_array, N))
    
    # plot results
    plt.plot(simulationTime, solutionState[:,0], label='S', color = 'blue')
    plt.plot(simulationTime, solutionState[:,1], label='E', color = 'red')
    plt.plot(simulationTime, solutionState[:,2], label='I', color = 'green')
    plt.xlabel("time")
    plt.ylabel('population')
    plt.show()

    return solutionState



for (i, val) in enumerate(nn):
    if isinstance(val, torch.nn.Linear):
        print(f"layer number: {i}, in: {val.in_features}, out: {val.out_features}")
        print(f"weights: {val.weight}")


