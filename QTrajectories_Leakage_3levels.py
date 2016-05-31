# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:26:41 2016

@author: Martijn Sol
"""

# Need:
# Hamiltonian
# F operators

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


T = 1.0                 # end time
Nt = 101                # Number of timesteps
t = np.linspace(0,T,Nt) # time
dt = T / (Nt - 1)       # size timestep
Nr = 100                # Number of runs

def TransposeH(M):
    # Calculates the Hermitian transpose of matrix/vector M
    HM = (M.transpose()).conj()
    return HM

Smin1 = np.reshape(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (3,3))
Smin2 = np.reshape(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), (3,3))

# Initial condition
phi0 = np.array([1.0, 0.0, 0.0], dtype = complex)

# Gate signal
tg = T          # Duration of gate pulse
width = 1.0     # Determines the shape of the gaussian pulse
delta1 = 0.0    
delta2 = -5.0
Delta = delta2 - 2 * delta1
lamb = np.sqrt(2)

# Gaussian
Eg = np.exp(-(t - tg / 2) ** 2 / (2 * width ** 2))
Eg = Eg - Eg[0]
Eg = Eg / sum(Eg*dt)
Eg = Eg * np.pi

# Choose pulse form
# Simple Gaussian 
Ex = Eg
Ey = np.zeros(Nt)

# DRAG
#Ex = Eg + (lamb ** 2 - 4) * Eg ** 3 / (8 * Delta ** 2)
#Ey = np.zeros(Nt)
#Ey[0:Nt-1] = -np.diff(Eg) / (Delta * dt)

# Hamiltonian

H0 = np.zeros((3,3), dtype = complex)
H0[1,1] = delta1
H0[2,2] = delta2

Hx = np.zeros((3,3), dtype = complex)
Hx[0,1] = 1
Hx[1,2] = lamb
Hy = 1j * Hx
Hx = (Hx + TransposeH(Hx))
Hy = (Hy + TransposeH(Hy))

# Jump operators
T11 = 2.0
F1HF1 = np.dot(TransposeH(Smin1), Smin1)
T12 = 2.0
F2HF2 = np.dot(TransposeH(Smin2), Smin2)


# Calculation
phi = np.zeros((3, Nt), dtype = complex)
phi[:,0] = phi0
#Dipole = np.zeros((Nt, Nr))
#DipoleSum = np.zeros((Nt, 1))
alpha2 = np.zeros((Nt,Nr))
beta2 = np.zeros((Nt,Nr))
gamma2 = np.zeros((Nt,Nr))
alpha2[0] = phi0[0]
beta2[0] = phi0[1]
gamma2[0] = phi0[2]

for k in range(Nr):
    for i in range(Nt-1): 
        # Construction of Hamitonian with the simple gaussian
        HR = H0 + 0.5 * Ex[i] * Hx + 0.5 * Ey[i] * Hy
        
        # Construction of Hamitonian with the DRAG method
#        HV = np.zeros((3,3), dtype = complex)
#        HV[0,1] = 0.5 * Ex[i]
#        HV[1,0] = 0.5 * Ex[i]
#        HV[2,0] = lamb * Ex[i] ** 2 / (8 * Delta)
#        HV[0,2] = lamb * Ex[i] ** 2 / (8 * Delta)
#        HV[2,2] = delta2 + (lamb ** 2 + 2) * Ex[i] ** 2 / (4 * Delta)
#        HR = HV        
#        
        # Effective Hamiltonian
        Heff = HR - .5j / T11 * F1HF1 - .5j / T12 * F2HF2
        U = sp.linalg.expm(-1j*dt*Heff)
        
        # Simulate Jumps
        p1 = np.dot(TransposeH(phi[:,i]), np.dot(F1HF1, phi[:,i])) * dt / T11
        p2 = np.dot(TransposeH(phi[:,i]), np.dot(F2HF2, phi[:,i])) * dt / T12
        p = p1 + p2
        #    print(p)
        
        phi[:,i+1] = np.dot(U, phi[:,i])
        norm = np.dot(TransposeH(phi[:,i+1]), phi[:,i+1])
        phi[:,i+1] = phi[:,i+1] / np.sqrt(norm)
        
        if p > np.random.random():
            ra = np.random.random()
            
            if ra < p1/p:
                phi[:,i+1] = np.dot(Smin1, phi[:,i])
                norm = np.dot(TransposeH(phi[:,i+1]), phi[:,i+1])
                phi[:,i+1] = phi[:,i+1] / np.sqrt(norm)
            if ra > p1/p:
                phi[:,i+1] = np.dot(Smin2, phi[:,i])
                norm = np.dot(TransposeH(phi[:,i+1]), phi[:,i+1])
                phi[:,i+1] = phi[:,i+1] / np.sqrt(norm)
        
#       The probabilities of being in a certain level
        alpha2[i+1,k] = (phi[:,i]*phi.conj()[:,i])[0] # The ground level
        beta2[i+1,k] = (phi[:,i]*phi.conj()[:,i])[1]  # First excited state
        gamma2[i+1,k] = (phi[:,i]*phi.conj()[:,i])[2] # Second exited state (leakage)
    
#    print(phi[:,i+1])

# Calculate average trajectory
alpha2Ave = np.mean(alpha2, axis = 1)
beta2Ave = np.mean(beta2, axis = 1)
gamma2Ave = np.mean(gamma2, axis = 1)

# Plot all trajectories
fig = plt.figure(figsize=(8,6))
plt.plot(t, alpha2 , 'b-', t, beta2, 'r', t, gamma2, 'g')
plt.ylabel('Prob',fontsize=18)
plt.xlabel('t',fontsize=18)
plt.axis([0, T, -0.1, 1.1])

plt.grid()
plt.show()

# Plot average trajectory
fig = plt.figure(figsize=(8,6))
line_a, = plt.plot(t, alpha2Ave , 'b') #, label='$\alpha$')
line_b, = plt.plot(t, beta2Ave, 'r') #, label='|$\beta$|^2')
line_c, = plt.plot(t, gamma2Ave, 'g') #, label='|$\gamma$|^2')
plt.legend([line_a, line_b, line_c], ['|alpha|^2', '|beta|^2', '|gamma|^2'])
plt.ylabel('Prob',fontsize=18)
plt.xlabel('t',fontsize=18)
plt.axis([0, T, -0.1, 1.1])
plt.grid()
plt.show()

print('The blue line is the probability to be in the ground state')
print('The red line is the probability to be in the first excited state')
print('The green line is the probability to be in the second excited (leakage) state')





