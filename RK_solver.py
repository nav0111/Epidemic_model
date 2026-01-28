import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import torch

# in ipython, run the following for interactive reloading
# %load_ext autoreload
# %autoreload 2
# import RK_solver


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

def add_noise(data, noise_level):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

"""def seirStateSpaceModel(y,t,timePoints, sigma0, gamma0, betaArray, N):
    S, E, I =y
    beta_t = np.interp(t, timePoints, betaArray)
    print(beta_t)
    #beta_t = betaArray[t]
    dSdt = -beta_t *S*I /N
    dEdt = beta_t *S *I /N - sigma0*E
    dIdt = sigma0*E - gamma0*I
    return [dSdt, dEdt, dIdt]"""

#SEIHRD model
def seirhrd_model(y,t,time_points, sigma0, gamma_u0, gamma_r0,p0,gamma_h0,mu0, beta_array, N):
    S, E, I_u, I_r, H, R, D =y
    beta_t = np.interp(t,time_points, beta_array)
    dSdt = -beta_t * S * (I_u + I_r) /N
    dEdt = beta_t * S * (I_u + I_r) /N - sigma0*E
    dI_udt = p0 * sigma0 * E - gamma_r0 * I_r
    dI_rdt = (1-p0) * sigma0 * E - gamma_h0 * I_r
    dHdt = gamma_h0 * I_r - gamma_u0 * H
    dRdt = gamma_u0 * H + gamma_r0 * I_r
    dDdt = mu0 * H
    return [dSdt, dEdt, dI_udt, dI_rdt, dHdt, dRdt, dDdt]

def run_seihrd_model(): 
    # define parameters
    N= 1000
    sigma0 = 1/5.2
    gamma_u0 = 1/10
    gamma_r0 = 1/7
    p0 = 0.8
    gamma_h0 = 1/14
    mu0 = 0.01
    startTime =0
    endTime = 100
    timeSteps = 1000
    simulationTime = np.linspace(startTime, endTime, timeSteps)

    beta_array = create_beta_array(simulationTime)
    
    # define initial conditions
    S0, E0, I_u0, I_r0, H0, R0, D0 = 990,5, 2,3, 0,0,0
     
    # run simulation
    initialState = np.array([S0, E0, I_u0, I_r0, H0, R0, D0])
    solutionState = odeint(seirhrd_model, initialState, simulationTime, args=(simulationTime, sigma0, gamma_u0, gamma_r0,p0,gamma_h0,mu0,beta_array,N))

    return solutionState

def plot_solution(solutionState):
    # if time start and end change, 
    # update here and in run_seihrd_model.
    ts = np.linspace(0, 100, solutionState.shape[0])
    # plot results
    plt.plot(ts, solutionState[:,0], label='S', color = 'blue')
    plt.plot(ts, solutionState[:,1], label='E', color = 'red')
    plt.plot(ts, solutionState[:,2], label='I_u', color = 'green')
    plt.plot(ts, solutionState[:,3], label ='I_r', color = 'orange')
    plt.plot(ts, solutionState[:,4], label='H', color = 'purple')
    plt.plot(ts, solutionState[:,5], label= 'R', color = 'brown')
    plt.plot(ts, solutionState[:,6], label= 'D', color = 'black')
    plt.xlabel("time")
    plt.ylabel('population')
    plt.legend()
    plt.show()
    plt.close()



