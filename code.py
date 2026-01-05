import numpy as np
import torch
import torch.nn as nn
import torch.autograd as grad
import torch.nn.functional as func
import matplotlib.pyplot as plt

#Initial Conditions
N = 1000
S0 = 990
E0 = 5
Iu0 = 2
Ir0 = 3
H0 = 0
R0 = 0
D0 = 0

#Model Parameters initial values
beta0 = 0.3    #transmission rate
sigma0 = 1/ 5.2 #incubation rate (1/latent period)
gamma_u0 = 1/10 #recovery rate from unreported people
gamma_r0 = 1/14 #recovery rate from reported people
p0 = 0.4        # proportion of exposed going to report
h0 = 0.05       #Hospitalization rate
gamma_h0 = 1/10 # recovery rate from hospitalized
mu0 = 0.02      # Death rate from hospitalized

#Time hprizon in days
T = 100 
def create_pinn_model():
    model = nn.Sequential(
        nn.Linear(1,50),
        nn.Tanh(),
        nn.Linear(50,50),
        nn.Tanh(),
        nn.Linear(50,50),
        nn.Tanh(),
        nn.Linear(50,8)  #return 7 states and beta
    )
    return model

#Loss functions
#ODE loss
def compute_ode_loss(model,t_col):
    """
    creates an independent variable t_col and get derivatives with respect to it
    t_col.clone() creates a new tensor in memory
    detach() is, cutting the tensor from any previous computation

    """
    t_col = t_col.clone().detach().requires_grad_(True)
    out = model(t_col)
    states = out[:, [0,1,2,3,4,5,6]] #Gives S, E, Iu, Ir, H, R, D
    raw_beta = out[:, 7]
    beta = torch.exp(raw_beta)  # ensuring beta > 0

    #compute derivatives Using autograd
    d_states_dt = []
    for i in range(7):
        grad_val = torch.autograd.grad(states[:,i].sum(), t_col, retain_graph = True, create_graph = True)[0]
        d_states_dt.append(grad_val.view(-1,1))
    d_states_dt = torch.cat(d_states_dt, dim=1) #creates a matrix contains all derivatives columnwise

    S, E, Iu, Ir, H, R, D = states[:,0], states[:,1], states[:,2], states[:,3], states[:,4], \
                                states[:,5], states[:,6]
    dSdt, dEdt, dIudt, dIrdt, dHdt, dRdt, dDdt = d_states_dt[:,0], d_states_dt[:,1], d_states_dt[:,2],\
                                                     d_states_dt[:,3], d_states_dt[:,4], d_states_dt[:,5],\
                                                     d_states_dt[:,6]
        
    #ODE residuals
    I_total = Iu + Ir    #Total infected people
    Transmission = beta * S * I_total

    res_S = dSdt + Transmission
    res_E = dEdt - (Transmission - sigma0 * E)
    res_Iu = dIudt - ((1-p0) * sigma0 * E - gamma_u0 * Iu)
    res_Ir = dIrdt - (p0 * sigma0 * E - gamma_r0 * Ir - h0 * Ir)
    res_H = dHdt - (h0 * Ir - gamma_h0 * H - mu0 * H)
    res_R = dRdt - (gamma_u0 * Iu + gamma_r0 * Ir + gamma_h0 * H)
    res_D = dDdt - (mu0 * H)

    ode_loss = torch.mean(res_S**2 + res_E**2 + res_Iu**2 + res_Ir**2 + res_H**2 +
                              res_R**2 + res_D**2)
        
    return ode_loss, beta
    

#Compute IC loss
def compute_ic_loss(model):
    #create a tensor with initial time t=0 and [[0]] means its a 2d tensor, and do not 
    #compute tensor wrt this tensor
    t_ic = torch.tensor([[0.0]], requires_grad = False)
    out_ic = model(t_ic)
    states_ic = out_ic[:, [0,1,2,3,4,5,6]].squeeze() #the row dimension of size 1 is removed

    S_ic, E_ic, Iu_ic, Ir_ic, H_ic, R_ic, D_ic = states_ic[0], states_ic[1], states_ic[2], \
                                                 states_ic[3], states_ic[4], states_ic[5],\
                                                 states_ic[6]
    
    loss = (S_ic - S0)**2 + (E_ic - E0)**2 + (Iu_ic - Iu0)**2 + (Ir_ic - Ir0)**2 + \
           (H_ic - H0)**2 + (R_ic - R0)**2 + (D_ic - D0)**2
    
    return loss

#Train the PINN model
def train_pinn():
    model = create_pinn_model()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    num_col = 2000     #collocation points
    t_col = torch.rand(num_col, 1) * T
    lambda_ic = 100    #weight for ic loss

    print("Training PINN:")
    for epochs in range(10000):
        optimizer.zero_grad()

        ode_loss, beta = compute_ode_loss(model, t_col)
        ic_loss = compute_ic_loss(model)
        total_loss = ode_loss + ic_loss

        total_loss.backward()
        optimizer.step()

        if epochs%1000 == 0:
            print(f"Epoch{epochs}: Total Loss = {total_loss.item(): .5f}, ODE Loss = {ode_loss.item():.5f},\
                  IC Loss = {ic_loss.item():.5f}")
            
    return model
    

#Evaluation and plotting
model = train_pinn()
t_test = torch.linspace(0,T,100).reshape(-1,1)

with torch.no_grad():
    out_test = model(t_test)
    states_test = out_test[:, 1:7].numpy()
    raw_beta_test = out_test[:,7].numpy()
    beta_test = np.exp(raw_beta_test)

t_numpy = t_test.numpy().flatten()

#Plot all the compartments
compartments = ['S (Susceptible)', 'E (Exposed)', 'Iu (Undetected Infectious)', 
                   'Ir (Reported Infectious)', 'H (Hospitalized)', 'R (Recovered)', 'D (Deaths)']
colors = ['blue', 'orange', 'red', 'purple', 'brown', 'green', 'black']

for i in range(7):
        plt.subplot(3, 3, i+1)
        plt.plot(t_numpy, states_test[:, i], color=colors[i], linewidth=2)
        plt.title(compartments[i])
        plt.xlabel('Time in days')
        plt.ylabel('Number of people')
        plt.grid(True)

 # Plot learned beta(t)
plt.subplot(3, 3, 8)
plt.plot(t_numpy, beta_test, color='red', linewidth=2)
plt.title('Beta(t) - Time-dependent Transmission Rate')
plt.xlabel('Time in Days')
plt.ylabel('Transmission Rate')
plt.grid(True)


        









                              
        






