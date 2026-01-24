# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as grad
import matplotlib.pyplot as plt
import RK_solver

# %%
#calling the data
#Only the reported infected cases
data = RK_solver.run_seihrd_model()
I_obs = data[:,3]
noisy_I_obs = RK_solver.add_noise(I_obs, noise_level= 0.2)

# PINN model
def pinn_model_one():
    model = nn.Sequential(
        nn.Linear(1, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 8)
    )
    return model

# Data loss function
def data_loss(model, t, I_obs):
    I_pred = model(t)[:, 3]
    I_obs_tensor = torch.tensor(I_obs, dtype=torch.float32)
    loss = nn.MSELoss()(I_pred, I_obs_tensor)
    return loss

# ODE loss function
def ode_loss(model, t, sigma0, gamma_u0, gamma_r0, p0, h0, gamma_h0, mu0, N):
    y_pred = model(t)
    
    # Extract state variables
    S = y_pred[:, 0]
    E = y_pred[:, 1]
    Iu = y_pred[:, 2]
    Ir = y_pred[:, 3]
    H = y_pred[:, 4]
    R = y_pred[:, 5]
    D = y_pred[:, 6]
    beta_t = torch.exp(y_pred[:, 7])  #for positive values
    
    # Compute derivatives
    dSdt = grad.grad(S, t, torch.ones_like(S), create_graph=True)[0]
    dEdt = grad.grad(E, t, torch.ones_like(E), create_graph=True)[0]
    dIudt = grad.grad(Iu, t, torch.ones_like(Iu), create_graph=True)[0]
    dIrdt = grad.grad(Ir, t, torch.ones_like(Ir), create_graph=True)[0]
    dHdt = grad.grad(H, t, torch.ones_like(H), create_graph=True)[0]
    dRdt = grad.grad(R, t, torch.ones_like(R), create_graph=True)[0]
    dDdt = grad.grad(D, t, torch.ones_like(D), create_graph=True)[0]
    
    # ODE residuals 
    I_total = Iu + Ir
    Transmission = beta_t * S * I_total
    
    res_S = dSdt + Transmission
    res_E = dEdt - (Transmission - sigma0 * E)
    res_Iu = dIudt - ((1-p0) * sigma0 * E - gamma_u0 * Iu)
    res_Ir = dIrdt - (p0 * sigma0 * E - gamma_r0 * Ir - h0 * Ir)
    res_H = dHdt - (h0 * Ir - gamma_h0 * H - mu0 * H)
    res_R = dRdt - (gamma_u0 * Iu + gamma_r0 * Ir + gamma_h0 * H)
    res_D = dDdt - (mu0 * H)
    
    # Total ODE loss
    loss = (res_S**2).mean() + (res_E**2).mean() + (res_Iu**2).mean() + \
           (res_Ir**2).mean() + (res_H**2).mean() + (res_R**2).mean() + \
           (res_D**2).mean()
    
    return loss

# Initial condition loss
def ic_loss(model, S0, E0, Iu0, Ir0, H0, R0, D0):
    t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=False)
    y0_pred = model(t0)
    
    S_ic = y0_pred[:, 0]
    E_ic = y0_pred[:, 1]
    Iu_ic = y0_pred[:, 2]
    Ir_ic = y0_pred[:, 3]
    H_ic = y0_pred[:, 4]
    R_ic = y0_pred[:, 5]
    D_ic = y0_pred[:, 6]
    
    loss = ((S_ic - S0)**2 + (E_ic - E0)**2 + (Iu_ic - Iu0)**2 + 
            (Ir_ic - Ir0)**2 + (H_ic - H0)**2 + (R_ic - R0)**2 + 
            (D_ic - D0)**2).mean()
    
    return loss

# Training function
def train_pinn_model_one():
    # Hyperparameters
    learning_rate = 0.001
    epochs = 2000
    N = 10000.0
    sigma0 = 1/5.2
    gamma_u0 = 1/14
    gamma_r0 = 1/14
    p0 = 0.8
    h0 = 0.05
    gamma_h0 = 1/21
    mu0 = 0.02
    
    # Initial conditions (normalized by N)
    S0 = (N - 100) / N
    E0 = 50 / N
    Iu0 = 20 / N
    Ir0 = 30 / N
    H0 = 0 / N
    R0 = 0 / N
    D0 = 0 / N
    
    # Time points
    t = np.linspace(0, 200, 1000)
    t_tensor = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
    t_tensor.requires_grad_(True)
    
    # Normalize observations
    I_obs_normalized = noisy_I_obs / N
    
    # Model
    mod = pinn_model_one()
    optimizer = optim.Adam(mod.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        loss_data = data_loss(mod, t_tensor, I_obs_normalized)
        loss_ode = ode_loss(mod, t_tensor, sigma0, gamma_u0, gamma_r0, 
                           p0, h0, gamma_h0, mu0, N)
        loss_ic = ic_loss(mod, S0, E0, Iu0, Ir0, H0, R0, D0)
        
        # Weighted combination
        total_loss = 100 * loss_data + 1 * loss_ode + 100 * loss_ic
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Total: {total_loss.item():.4f}, "
                  f"Data: {loss_data.item():.4f}, "
                  f"ODE: {loss_ode.item():.4f}, "
                  f"IC: {loss_ic.item():.4f}")
    
    return mod

# predict and plot results
def plot_results(model,t, N):
    t_tensor = torch.tensor(t, dtype=torch.float32) .reshape(-1,1)
    #Extracting the states
    with torch.no_grad():
        y_pred = model(t_tensor)
        #Extracting the states and scaling back to population
        S_pred = y_pred[:, 0] * N
        E_pred = y_pred[:, 1] * N
        Iu_pred = y_pred[:, 2] * N
        Ir_pred = y_pred[:, 3] * N
        H_pred = y_pred[:, 4] * N
        R_pred = y_pred[:, 5] * N
        D_pred = y_pred[:, 6] * N
        beta_t = torch.exp(y_pred[:, 7])

    #plotting
    preds = [S_pred, E_pred, Iu_pred, Ir_pred, H_pred, R_pred, D_pred,
             beta_t]
    labels = ['Susceptibles', 'Exposed', 'Infected Unreported', 'Infected Reported',
              'Hospitalized', 'Recovered', 'Dead', 'Transmission Rate']
    
    plt.figure(figsize=(15,10))
    for i in range(len(preds)):
        plt.subplot(3,3,i+1)
        plt.plot(t, preds[i], label = labels[i])
        if labels[i] == 'Infected Reported':
            plt.plot(t, noisy_I_obs, label = "true Infected Reported", color =
                     'red')
            plt.legend()
            plt.title(f"f{labels[i]} Predicted vs True")
        else:
            plt.title(f"{labels[i]} Predicted")
            plt.legend()
            plt.xlabel("Time in Days")
            plt.ylabel("Population")
            plt.grid(True)
    plt.show()
    return preds, labels

    
    
