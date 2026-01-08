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

    #Extracting initial values
    S_ic = states_ic[0]
    E_ic = states_ic[1]
    Iu_ic = states_ic[2]
    Ir_ic = states_ic[3]
    H_ic = states_ic[4]
    R_ic = states_ic[5]
    D_ic = states_ic[6]

    ic_loss = ((S_ic - S0)**2 + (E_ic - E0)**2 + (Iu_ic - Iu0)**2 + (Ir_ic - Ir0)**2 + 
               (H_ic - H0)**2 + (R_ic - R0)**2 + (D_ic - D0)**2)/7
    return ic_loss

#ODE loss for model 2
def ode_loss2(model, t_col, beta_min, beta_max, a, b):
    t_col = t_col.clone().detach().requires_grad_(True)
    out = model(t_col)
    states = out[:, :7]
    beta = compute_sigmoid_beta(t_col, beta_min, beta_max, a, b)

    #derivatives
    d_states_dt = []
    for i in range(7):
        grad_val = torch.autograd.grad(states[: ,i].sum(), t_col, retain_graph = True, create_graph = True)[0]
        d_states_dt.append(grad_val.view(-1,1))
    d_states_dt = torch.cat(d_states_dt , dim =1)

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
    Transmission = beta.squeeze() * S * I_total

    #ODE Residuals
    res_S = dSdt + Transmission
    res_E = dEdt - (Transmission - sigma0 * E)
    res_Iu = dIudt - ((1-p0) * sigma0 * E - gamma_u0 * Iu)
    res_Ir = dIrdt - (p0 * sigma0 * E - gamma_r0 * Ir - h0*Ir)
    res_H = dHdt - (h0 * Ir - gamma_h0 * H - mu0* H)
    res_R = dRdt - (gamma_u0 * Iu + gamma_r0 *Ir + gamma_h0 * H)
    res_D = dDdt - (mu0 * H)

    ode_loss = torch.mean(res_S**2 + res_E**2 + res_Iu**2 + res_Ir**2 + res_H**2 + res_R**2+ res_D**2)
    total_pop = S + E + Iu + Ir + H + R + D
    con_loss = torch.mean((total_pop - 1)**2)

    return ode_loss, con_loss, beta

#IC loss for model 2
def ic_loss2(model, beta_min, beta_max, a, b):
    t_ic = torch.tensor([[0.0]], requires_grad = False)
    out = model(t_ic)
    states_ic = out[:, 7].squeeze()
    #Extract initial values
    S_ic = states_ic[0]
    E_ic = states_ic[1]
    Iu_ic = states_ic[2]
    Ir_ic = states_ic[3]
    H_ic = states_ic[4]
    R_ic = states_ic[5]
    D_ic = states_ic[6]

    #initial condition loss
    ic_loss = ((S_ic - S0)**2 + (E_ic - E0)**2 + (Iu_ic - Iu0)**2 + (Ir_ic - Ir0)**2 + (H_ic - H0)**2 + (R_ic - R0)**2 + (D_ic - D0)**2)
    return ic_loss

#function for gradient magnitude
def compute_grad_magnitude(loss, parameters):
    """
       find the magnitude of gradients by using L2 norm wrt parameters
       L2 norm: sqrt(grad[0]^2 + grad[1]^2 + ... + grad[n]^2)
       First making sure that the parameters is a list
       then compute gradients, some parameters might be None if parameter unused
    """
    params = list(parameters)
    grads = torch.autograd.grad(loss, params, retain_graph = True, create_graph = False)
    grad_norms = []
    for grad in grads:
        if grad is not None:
            norm_val = torch.norm(grad, p=2)
            grad_norms.append(norm_val)

    #No gradients computed at the boundary
    #no parameter was affected by this loss
    if len(grad_norms) == 0:
        return torch.tensor(0.0)
    
    #Average all the gradient loss, torch.stack combine the list into a tensor
    avg_magnitude = torch.mean(torch.stack(grad_norms))
    return avg_magnitude

#Function of smart weighting system
def compute_smart_weights(ode_loss, ic_loss, con_loss, parameters, prev_weights, epoch, total_epochs, epsilon = 1e-8):
    """
    smart weighting system with bounds where we considered loss value
    Args:
        prev_weights: weights from previous epoch
    """
    #compute gradient magnitudes
    grad_mag_ode = compute_grad_magnitude(ode_loss, parameters)
    grad_mag_ic = compute_grad_magnitude(ic_loss, parameters)
    grad_mag_con = compute_grad_magnitude(con_loss, parameters)

    #Get loss values
    loss_val_ode = ode_loss.item()
    loss_val_ic = ic_loss.item()
    loss_val_con = con_loss.item()

    #Here alpha is adaptive smoothing factor, controls how fast weights change
    #Early epochs, alpha =0.3 for fast adaption, alpha =0.05, late epochs for slow adaption
    #How quickly the loss weights change because of new information
    progress = epoch / total_epochs #range [0,1], normalized time variable
    alpha = 0.3 * (1-progress) + 0.05 * progress

    #Compute average gradient magnitudfe
    avg_grad_mag = (grad_mag_ode + grad_mag_ic + grad_mag_con) / 3

    #Compute raw_weights based on gradients and loss value
    #The idea is if a loss has big gradients, give it a smaller weight. and if it has small
    #gradient then give it a bigger weight
    #Adding 0.1*loss_ode to prevent loss explosion (avg_grad/ grad =inf if grad=0)
    weight_ode = avg_grad_mag / (torch.max(grad_mag_ode, torch.tensor(epsilon)) +0.1*loss_val_ode)
    weight_ic = avg_grad_mag / (torch.max(grad_mag_ic, torch.tensor(epsilon)) + 0.1*loss_val_ic)
    weight_con = avg_grad_mag / (torch.max(grad_mag_con, torch.tensor(epsilon)) + 0.1*loss_val_con)

    #Apply bounds to prevent extreme values
    #Ode: max =10 to prevent domination
    #IC: min 10, max 100 
    #Conservation: min 5 max 50
    #.clamp is forcing to stay in the range
    weight_ode_b = torch.clamp(weight_ode, min =0.5, max =10).item()
    weight_ic_b = torch.clamp(weight_ic, min =10, max=100).item()
    weight_con_b = torch.clamp(weight_con, min = 5, max =50).item()

    #Now we smooth the weights using exponential moving average
    #EMA prevents weights from reacting too violently to noisy gardients,and ensures stable and convergent pinn training
    if prev_weights is None:
        w_ode = weight_ode_b
        w_ic = weight_ic_b
        w_con = weight_con_b
    else:
        w_ode = alpha * weight_ode_b + (1-alpha) * prev_weights[0]
        w_ic = alpha * weight_ic_b + (1-alpha) * prev_weights[1]
        w_con = alpha * weight_con_b + (1-alpha) * prev_weights[2]

    return (w_ode, w_ic, weight_con), (grad_mag_ode.item(), grad_mag_ic.item(), grad_mag_con,item())

#Training model 1
def train_model1():
    print("Training Model 1(a): Beta(t) from PINN")
    model = create_pinn_one_model()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    num_col = 2000
    t_col = torch.rand(num_col,1) *T

    prev_weights = None
    weight_hist = {}












    



    







