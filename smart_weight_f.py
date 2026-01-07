import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as grad
import matplotlib.pyplot as plt

#Initial Conditions, normalized by N
N = 1000
S0, E0, Iu0, Ir0, H0, R0, D0 = 990/N, 5/N, 3/N, 2/N, 0/N, 0/N, 0/N

#Initial Values for the parameters
#sigma0 is incubation rate at t=0, (1/latent period)
#Gamma_u0,_r0 recovery rate from unreported and reported people
#Proportion of exposed people going to report
#h0 is hospitalization rate
#gamma_h0 is rate of recovery after hospitalization
#mu0 is rate of death after hospitalized
sigma0, gamma_u0, gamma_r0, p0, h0, gamma_h0, mu0 = 1/5.2, 1/10, 1/14, 0.4, 0.05, 1/10, 0.02

#Time horizon in days
T = 200

#Model 1 where Beta(t) is estimated from PINN
def create_pinn_one_model():
    model = nn.Sequential(
        nn.Linear(1,50),
        nn.Tanh(),
        nn.Linear(50,50),
        nn.Tanh(),
        nn.Linear(50,50)
        nn.Tanh(),
        nn.Linear(50,8)

    )
    return model

#Model 2 where Beta(t) is a fixed sigmoid function
def create_pinn_two_model():
    model = nn.Sequential(
        nn.Linear(1,50),
        nn.Tanh(),
        nn.Linear(50,50),
        nn.Tanh(),
        nn.Linear(50,50),
        nn.Tanh(),
        nn.Linear(50,8)
    )
    return model

#Function of computing sigmoid Beta(t)
def compute_sigmoid_beta(t, beta_min, beta_max, a, b):
    """
    beta(t) = beta_min + (beta_max - beta_min) * sigmoid(a*t + b)
    beta_min, beta_max are the min and max transmission rate
    a is steepness parameter abd b is shift parameter
    """
    sigmoid_val = torch.sigmoid(a*t + b)
    beta = beta_min + (beta_max - beta_min) * sigmoid_val
    return beta

#computaion of loss functions
#ODE Loss for Model 1 
def ode_loss1(model, t_col):
    """
    Make a copy of collocation points of time t, for computing gradients
    Then extract the compartments and beta , make sure beta is positive
    Then find the derivatives of the compartments wrt t_col

    """
    t_col = t_col.clone().detach().requires_grad_(True)
    out = create_pinn_one_model(t_col)
    states = out[:, :7]
    raw_beta = out[:, 7]
    beta = torch.exp(raw_beta)
    
    d_states_dt = []
    for i in range(7):
        grad_val = torch.autograd.grad(states[:,i].sum(), t_col, retain_graph = True, 
                                       create_graph = True)[0]
        d_states_dt.append(grad_val.view(-1,1))
    d_states_dt = torch.cat(d_states_dt, dim=1)

    #Extract the values for each compartments
    S = states[:,0]
    E = states[:,1]
    Iu = states[:,2]
    Ir = states[:,3]
    H = states[:,4]
    R = states[:,5]
    D = states[:,6]

    #Extract the derivalive values
    dSdt = d_states_dt[:,0]
    dEdt = d_states_dt[:,1]
    dIudt = d_states_dt[:,2]
    dIrdt = d_states_dt[:,3]
    dHdt = d_states_dt[:,4]
    dRdt = d_states_dt[:,5]
    dDdt = d_states_dt[:,6]

    I_total = Iu + Ir
    Transmission = beta.squeeze() * S * I_total #.squeeze: beta is just a list of beta values (flat line), not a column vector

    #Residuals
    res_S = dSdt + Transmission
    res_E = dEdt - (Transmission - sigma0 * E)
    res_Iu = dIudt - ((1-p0) * sigma0 * E - gamma_u0 * Iu)
    res_Ir = dIrdt - (p0 * sigma0 * E - gamma_r0 * Ir - h0 * Ir)
    res_H = dHdt - (h0 * Ir - gamma_h0 * H - mu0 * H)
    res_R = dRdt - (gamma_u0 * Iu + gamma_r0 * Ir + gamma_h0 * H)
    res_D = dDdt - (mu0 * H)

    #ODE Loss
    ode_loss = torch.mean(res_S**2 + res_E**2 + res_Iu**2 + res_Ir**2 + res_H**2 + res_R**2 + res_D**2)

    total_pop = S + E + Iu + Ir + H + R + D
    con_loss = torch.mean((total_pop - 1)**2)

    return ode_loss, con_loss, beta

#Initial condition loss
def ic_loss1(model):
    """
    Create a tensor with initial time point t= 0, [[0]] is a 2_d vector 
    and do not compute derivative wrt this tensor
    """
    t_ic = torch.tensor([[0.0]], requires_grad = False)
    out_ic = model(t_ic)
    states_ic = out_ic[:, :7].squeeze()
    







