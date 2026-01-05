import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#Initial conditions (Normalized by N)
N = 1000
S0 = 990/N
E0 = 5/N
Iu0 = 2/N
Ir0 = 3/N
H0 = 0/N
R0 = 0/N
D0 = 0/N

#Initial Values of Model Parameters
sigma0 = 1/5.2     #incubation rate (1/latent period)
gamma_u0 = 1/10    #recovery rate from unreported people
gamma_r0 = 1/14    #recovery rate from reported people
p0 = 0.4           #proportion of exposed people going to report
h0 = 0.05          #Hospitalization rate
gamma_h0 = 1/10    #recovery rate from hospitalization
mu0 = 0.02         #Death rate after hospitalized

#Time horizon in days
T = 100

#Model 1(a), where we estimate beta(t) from PINN
def create_pinn_model():
    """
    A fully connected NN, where the input is time 't', and it returns 8 values (7 compartments and beta value)
    It has three hidden layers each with 50 neurons
    """
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

#Model 1(b), where we assume that beta(t) is a sigmoid function
def create_sigmoid_beta_model():
    """
    Returns 7 compartments
    """
    model = nn.Sequential(
        nn.Linear(1,50),
        nn.Tanh(),
        nn.Linear(50,50),
        nn.Tanh(),
        nn.Linear(50,50),
        nn.Tanh(),
        nn.Linear(50,7)
    )
    return model

#Learnable Parameters for Sigmoid beta, initial assumptions
#min , and max transmission rate. a is steepness parameter and b is shift parameter
beta_min = torch.tensor([0.05], requires_grad = True)      
beta_max = torch.tensor([0.5], requires_grad = True)
a = torch.tensor([0.1], requires_grad = True)  
b = torch.tensor([0.0], requires_grad = True)

def compute_sigmoid_beta(t, beta_min, beta_max, a, b):
    """
   beta(t) = beta_min + (beta_max - beta_min) * sigmoid(a*t + b)
    """
    sigmoid_val = torch.sigmoid(a*t +b)
    beta = beta_min + (beta_max - beta_min) * sigmoid_val
    return beta

#Loss Functions for model 1(a)
#ODE loss where the model doesnot return beta
def compute_ode_loss(model,t_col):
    """
    first make a copy of the collocation points of time 't' so that we can compute gradients with respect to it
    Then extract the compartments and beta, we make sure beta is positive
    After that compute derivatives wrt t_col
    Loss is ODE residuals which should be zero
    """
    #model output
    t_col = t_col.clone().detach().requires_grad_(True)
    out = model(t_col)
    states = out[:, :7]  #First 7 columns (S,E,Iu,Ir,H,R,D)
    raw_beta = out[:, 7] #8th column for beta
    beta = torch.exp(raw_beta)

    #compute derivatives of states
    d_states_dt = []
    for i in range(7):
        grad_val = torch.autograd.grad(states[:, i].sum(), t_col, retain_graph = True, create_graph = True)[0]
        d_states_dt.append(grad_val.view(-1,1))
    d_states_dt = torch.cat(d_states_dt, dim =1)

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
    Transmission = beta.squeeze() * S * I_total #now beta is just a list of beta values (flat line), not a column vector

    #ODE residuals
    res_S = dSdt + Transmission
    res_E = dEdt - (Transmission - sigma0 * E)
    res_Iu = dIudt - ((1-p0) * sigma0 * E - gamma_u0 * Iu)
    res_Ir = dIrdt - (p0 * sigma0 * E - gamma_r0 * Ir - h0 * Ir)
    res_H = dHdt - (h0 * Ir - gamma_h0 * H - mu0 * H)
    res_R = dRdt - (gamma_u0 * Iu + gamma_r0 * Ir + gamma_h0 * H)
    res_D = dDdt - (mu0 * H)

    #ODE Loss
    ode_loss = torch.mean(res_S**2 + res_E**2 + res_Iu**2 + res_Ir**2 + res_H**2 + res_R**2 + res_D**2)

    #Conservation of the compartments as we normalized by N
    total_pop = S + E + Iu + Ir + H + R + D
    conservation_loss = torch.mean((total_pop - 1)**2)

    return ode_loss, conservation_loss, beta

#Compute IC Loss
def compute_ic_loss(model):
    #create a tensor with initial time t=0 and [[0]] means its a 2d tensor, and do not 
    #compute derivatives wrt this tensor
    t_ic = torch.tensor([[0.0]], requires_grad = False)
    out_ic = model(t_ic)
    states_ic = out_ic[:, :7].squeeze()

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


#Loss functions for model 1(b), sigmoid beta
def compute_ode_loss_sigmoid_beta(model, t_col, beta_min, beta_max, a, b):
    t_col = t_col.clone().detach().requires_grad_(True)
    out = model(t_col)
    states = out[:, :7]

    #compute beta using sigmoid formula
    beta = compute_sigmoid_beta(t_col, beta_min, beta_max, a, b)

    #derivatives
    d_states_dt = []
    for i in range(7):
        grad_val = torch.autograd.grad(states[:, i].sum(), t_col, retain_graph = True, create_graph = True)[0]
        d_states_dt.append(grad_val.view(-1,1))
    d_states_dt = torch.cat(d_states_dt, dim =1)

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
    Transmission = beta.squeeze() * S * I_total #now beta is just a list of beta values (flat line), not a column vector

    #ODE residuals
    res_S = dSdt + Transmission
    res_E = dEdt - (Transmission - sigma0 * E)
    res_Iu = dIudt - ((1-p0) * sigma0 * E - gamma_u0 * Iu)
    res_Ir = dIrdt - (p0 * sigma0 * E - gamma_r0 * Ir - h0 * Ir)
    res_H = dHdt - (h0 * Ir - gamma_h0 * H - mu0 * H)
    res_R = dRdt - (gamma_u0 * Iu + gamma_r0 * Ir + gamma_h0 * H)
    res_D = dDdt - (mu0 * H)

    #ODE Loss
    ode_loss = torch.mean(res_S**2 + res_E**2 + res_Iu**2 + res_Ir**2 + res_H**2 + res_R**2 + res_D**2)

    #Conservation of the compartments as we normalized by N
    total_pop = S + E + Iu + Ir + H + R + D
    conservation_loss = torch.mean((total_pop - 1)**2)

    return ode_loss, conservation_loss, beta

#Compute initial condition loss for sigmoid beta
def compute_ic_loss_sigmoid_beta(model, beta_min, beta_max, a, b):
    #Initial time t=0
    t_ic = torch.tensor([[0.0]], requires_grad = False)
    out_ic = model(t_ic)
    states_ic = out_ic[:, :7].squeeze()

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

#Training model 1(a)
def train_model_1a():
    print("Training Model 1(a): Beta(t) from PINN")
    model = create_pinn_model()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    #collocation points
    #creates 2000 random numbers between[0,1], then multiply by 100, so we get random days between 0 and 100
    num_col = 2000
    t_col = torch.rand(num_col, 1) * T  

    #Loss weights
    Lambda_ic = 100
    lambda_con = 10

    for epoch in range(10000):
        optimizer.zero_grad()  #clears gradients from previous training steps, just keep the new
        ode_loss, conservation_loss, beta = compute_ode_loss(model, t_col)
        ic_loss = compute_ic_loss(model)
        total_loss = ode_loss + Lambda_ic * ic_loss + lambda_con * conservation_loss
        total_loss.backward()   #Compute gradients
        optimizer.step()        #update parameters

        if epoch % 2000 ==0:
            print(f"Epoch {epoch}: Loss={total_loss.item():.5f}, ODE={ode_loss.item():.5f}, IC={ic_loss.item():.5f}")
    return model

#Training Model 1(b), sigmoid beta(t)
def train_model_1b():
    print("Training Model 1(b): Beta(t) for sigmoid Function")
    model = create_sigmoid_beta_model()
    #The initial assumptions
    beta_min = torch.tensor([0.05], requires_grad = True)      
    beta_max = torch.tensor([0.5], requires_grad = True)
    a = torch.tensor([0.1], requires_grad = True)  
    b = torch.tensor([0.0], requires_grad = True)

    optimizer = torch.optim.Adam(list(model.parameters()) + [beta_min, beta_max, a, b], lr = 0.001)

    #collocation points
    num_col = 2000
    t_col = torch.rand(num_col,1) * T

    #Loss weights
    lambda_ic = 100
    lambda_con = 10

    for epoch in range(10000):
        optimizer.zero_grad()
        ode_loss, conservation_loss, beta = compute_ode_loss_sigmoid_beta(model, t_col, beta_min, beta_max, a, b)
        ic_loss = compute_ic_loss_sigmoid_beta(model, beta_min, beta_max, a, b)
        total_loss = ode_loss + lambda_ic * ic_loss + lambda_con * conservation_loss
        total_loss.backward()
        optimizer.step()

        if epoch % 2000 ==0:
            print(f"Epoch{epoch}: Loss={total_loss.item():.5f}, ODE={ode_loss.item():.5f}, IC={ic_loss.item():.5f}")
    return model, beta_min, beta_max, a, b

#Train the models
model_pinn = train_model_1a()
model_sigmoid, beta_min_new, beta_max_new, a_new, b_new = train_model_1b()

#Testing and Plotting
t_test = torch.linspace(0,T,200).reshape(-1,1)

#Geting Predictions from model 1(a)
#no_grad(): do not compute or store any gradients inside the bloch
with torch.no_grad():
    out_1a = model_pinn(t_test)
    states_1a = out_1a[:, :7].numpy() * N   #multiplication by N , returns the actual values
    raw_beta_1a = out_1a[:, 7].numpy()
    beta_1a = np.exp(raw_beta_1a)

#Getting Predictions from model 1(b)
with torch.no_grad():
    out_sigmoid = model_sigmoid(t_test)
    states_sigmoid = out_sigmoid[:, :7].numpy() * N
    beta_sigmoid = compute_sigmoid_beta(t_test, beta_min_new, beta_max_new, a_new, b_new).numpy()

t_numpy = t_test.numpy().flatten()

#Extract individual compartments
#From model 1(a)
S_pinn = states_1a[:,0]
E_pinn = states_1a[:,1]
Iu_pinn = states_1a[:,2]
Ir_pinn = states_1a[:, 3]
H_pinn = states_1a[:, 4]
R_pinn = states_1a[:, 5]
D_pinn = states_1a[:, 6]

#From Model 1(b)
S_sigmoid = states_sigmoid[:,0]
E_sigmoid = states_sigmoid[:,1]
Iu_sigmoid = states_sigmoid[:,2]
Ir_sigmoid = states_sigmoid[:, 3]
H_sigmoid = states_sigmoid[:, 4]
R_sigmoid = states_sigmoid[:, 5]
D_sigmoid = states_sigmoid[:, 6]

#Plot Susceptible
plt.figure(figsize=(10,6))
plt.plot(t_numpy, S_pinn, label = "from Beta(t) with pinn", linewidth = 2.5, color = 'blue')
plt.plot(t_numpy, S_sigmoid, label = "from Beta(t) with sigmoid", linewidth = 2.5, color = 'red')
plt.title("Susceptible Population")
plt.legend()
plt.xlabel("Days")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()

#Plot Exposed
plt.figure(figsize=(10,6))
plt.plot(t_numpy, E_pinn, label = "from Beta(t) with pinn", linewidth = 2.5, color = 'blue')
plt.plot(t_numpy, E_sigmoid, label = "from Beta(t) with sigmoid", linewidth = 2.5, color = 'red')
plt.title("Exposed Population")
plt.legend()
plt.xlabel("Days")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()

#Unreported Infected population
plt.figure(figsize=(10,6))
plt.plot(t_numpy, Iu_pinn, label = "from Beta(t) with pinn", linewidth = 2.5, color = 'blue')
plt.plot(t_numpy, Iu_sigmoid, label = "from Beta(t) with sigmoid", linewidth = 2.5, color = 'red')
plt.title("Unreported Infected Population")
plt.legend()
plt.xlabel("Days")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()

#Reported Infected Population
plt.figure(figsize=(10,6))
plt.plot(t_numpy, Ir_pinn, label = "from Beta(t) with pinn", linewidth = 2.5, color = 'blue')
plt.plot(t_numpy, Ir_sigmoid, label = "from Beta(t) with sigmoid", linewidth = 2.5, color = 'red')
plt.title("Reported Infected Population")
plt.legend()
plt.xlabel("Days")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()

#Hospitalized Population
plt.figure(figsize=(10,6))
plt.plot(t_numpy, H_pinn, label = "from Beta(t) with pinn", linewidth = 2.5, color = 'blue')
plt.plot(t_numpy, H_sigmoid, label = "from Beta(t) with sigmoid", linewidth = 2.5, color = 'red')
plt.title("Hospitalized Population")
plt.legend()
plt.xlabel("Days")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()

#Recovered Population
plt.figure(figsize=(10,6))
plt.plot(t_numpy, R_pinn, label = "from Beta(t) with pinn", linewidth = 2.5, color = 'blue')
plt.plot(t_numpy, R_sigmoid, label = "from Beta(t) with sigmoid", linewidth = 2.5, color = 'red')
plt.title("Recovered Population")
plt.legend()
plt.xlabel("Days")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()

#Deaths
plt.figure(figsize=(10,6))
plt.plot(t_numpy, D_pinn, label = "from Beta(t) with pinn", linewidth = 2.5, color = 'blue')
plt.plot(t_numpy, D_sigmoid, label = "from Beta(t) with sigmoid", linewidth = 2.5, color = 'red')
plt.title("Dead Population")
plt.legend()
plt.xlabel("Days")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()

#Beta comparison
plt.figure(figsize=(10,6))
plt.plot(t_numpy, beta_1a, label = "from Beta(t) with pinn", linewidth = 2.5, color = 'yellow')
plt.plot(t_numpy, beta_sigmoid, label = "from Beta(t) with sigmoid", linewidth = 2.5, linestyle = '--', color = 'green')
plt.title("Transmission Rate comparison")
plt.legend()
plt.xlabel("Days")
plt.ylabel("Transmission Rate")
plt.grid(True)
plt.show()

#Total Population
plt.figure(figsize=(10,6))
total_pinn = states_1a.sum(axis=1)
total_sigmoid = states_sigmoid.sum(axis=1)
plt.plot(t_numpy, total_pinn, label= "Beta(t) from PINN", linewidth = 2.5, color = 'grey')
plt.plot(t_numpy, total_sigmoid, label = "sigmoid Beta(t)", linewidth = 2.5, color = 'pink')
plt.title("Total population check")
plt.legend()
plt.xlabel("Days")
plt.ylabel("Total Population")
plt.show()

print("Model:1(a)")
print(f"Beta range: {beta_1a.min():.4f} to {beta_1a.max():.4f}")
print(f"Mean beta: {beta_1a.mean():.4f}")

print("Model:1(b)")
print(f"Beta range: {beta_sigmoid.min():.4f} to {beta_sigmoid.max():.4f}")
print(f"Mean beta: {beta_sigmoid.mean():.4f}")

print("Learned Sigmoid Parameters")
print(f"Beta_min={beta_min_new.item():.4f}")
print(f"Beta_max={beta_max_new.item():.4f}")
print(f"a={a_new.item():.4f}")
print(f"b={b_new.item():.4f}")
