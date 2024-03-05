

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.signal import find_peaks, welch
from utils import *

## ----------------------------------------------

## NATURAL FREQUENCY & MODE SHAPE CALCULATION (2-3)

## ----------------------------------------------

# Structural properties of the beam
E = 200e9  # Elasticity modulus (Pa)
L = 5  # Length (m)
d = 30e-2  # Cross-section diameter (m)
rho = 7.8e3  # Density (kg/m^3)
A = np.pi * (d / 2) ** 2  # Cross-sectional area (m^2)

# FEM definition
n = 3  # Number of elements

# Individual element stiffness and mass
l = L / n
k_element = E * A / l * np.array([[1, -1], [-1, 1]])
m_element = (rho * A * l) / 2 * np.eye(2)

# Construct the mass and stiffness matrices
K_global = np.zeros((n + 2, n + 2))
M_global = np.zeros((n + 2, n + 2))

# Loop from 1 to n+1
for i in range(1, n + 2):
    indices = np.arange(i - 1, i + 1)
    K_global[indices[:, np.newaxis], indices] += k_element
    M_global[indices[:, np.newaxis], indices] += m_element

# Apply boundary conditions - fixed-free
K_global = K_global[1:n + 1, 1:n + 1]
M_global = M_global[1:n + 1, 1:n + 1]
K_global[n-1, n-1] = K_global[n-1, n-1] / 2

# Damping matrix
alpha = 1e-5
beta = 1e-2
C_global = alpha * K_global + beta * M_global

# Force input matrix
force = np.zeros(n)
force[2] = 1

# Perform modal analysis - undamped system
eigenvalues, eigenvectors = la.eigh(K_global, M_global)

# Extract natural frequencies
omegaO = np.sqrt(eigenvalues.real)

# Psi contains the eigenvectors as columns
Psi = eigenvectors

# Modal damping
zeta = np.diag(Psi.T.dot(C_global).dot(Psi)) / (2 * omegaO)

# Plot natural frequencies
plt.figure(figsize=(5, 5))
plt.plot(np.arange(1, n + 1), omegaO / (2 * np.pi), 'o', color='black')
plt.grid(True)
plt.xlabel('Mode')
plt.ylabel('Frequency (Hz)')
plt.xlim(0.5, n + 0.5)
plt.xticks(np.arange(1, n + 1))
plt.title('Natural frequencies')
plt.show()

# Plot mode shapes
plt.subplots(n, 1, sharex=True, figsize=(10, 50))
for i in range(1, n+1):
    plt.subplot(n, 1, i)
    x_values = np.concatenate(([0], np.arange(1, n + 1) * l))
    y_values = np.concatenate(([0], Psi[:, i - 1]))
    
    plt.plot(x_values, y_values, '-o', color='black', markeredgecolor='black', markerfacecolor='white')
    plt.grid(True)
    if i == n: 
        plt.xlabel('Length (m)')
    plt.ylabel('Displacement')
    plt.xlim(0, 5.1)
    plt.title(f"Mode {i} - Frequency {omegaO[i - 1] / (2 * np.pi):.2f} Hz - Damping {100 * zeta[i - 1]:.2f}%")

plt.show()

## ----------------------------------------------

## FRF CALCULATION (4)

## ----------------------------------------------

# Define the frequency vector for FRF calculation
nf = 2501  # Number of frequency points
fs = 2500  # Sampling frequency
frequencies = np.linspace(0, fs/2, nf)
Omega = 2 * np.pi * frequencies

# Building the FRF matrix
Ho = np.zeros((n, nf), dtype=complex)
W_n = np.zeros([3, 3], dtype=object)

# Calculate FRF
for i in range(nf):
    A_1 = K_global - M_global * Omega[i]**2 + 1j * C_global * Omega[i]
    Ho[:, i] = la.solve(A_1, force)

# Find peaks
for i in range(3):
    peaks, _ = find_peaks(np.abs(Ho[i, :]))
    W_n[i, 0] = peaks
    W_n[i, 1] = np.take(frequencies, W_n[i, 0])
    W_n[i, 2] = np.take(np.abs(Ho[i, :]), W_n[i, 0]) 

# Plot FRF
plt.subplots(n, 1, sharex=True, figsize=(10, 50))
for i in range(n):
    plt.subplot(n, 1, i + 1)
    plt.semilogy(frequencies, np.abs(Ho[i, :]), color='black')
    plt.plot(W_n[i, 1], W_n[i, 2], 'o', markeredgecolor='black', markerfacecolor='white')
    plt.grid(True)
    if i == n - 1:
        plt.xlabel('Frequency (Hz)')
    plt.ylabel('FRF magnitude')
    plt.title(f'FRF DOF {i + 1}')
    plt.xlim(0, fs/2)

plt.show()
    
## ----------------------------------------------

## DAMPING CALCULATION - HALF POWER METHOD (5)

## ----------------------------------------------

# Calculate damping
damping = np.zeros([3, 3])
for i in range(3):  
    for j in range(3):
        for k in [-1, 1]:
            boundary = int(W_n[i, 0][j] + k * 50)
            if k == -1:
                vals = np.abs(Ho[i, boundary:int(W_n[i, 0][j])])
            else:
                vals = np.abs(Ho[i, int(W_n[i, 0][j]):boundary])
            
            half_peak = vals[np.abs(vals - W_n[i, 2][j]/np.sqrt(2)).argmin()]
            freq_at_half_peak = np.take(frequencies, np.where(vals == half_peak)[0] + (boundary if k == -1 else int(W_n[i, 0][j])))
            if k == -1:
                W_a1 = freq_at_half_peak
            else:
                W_a2 = freq_at_half_peak

        damping[i, j] = np.abs(W_a1 - W_a2) / (2 * W_n[i, 1][j]) * 100
        
## ----------------------------------------------

## STATE SPACE FORM - EXPONENTIAL METHOD (6)

## ----------------------------------------------

one = np.concatenate((np.zeros([3,3]),np.eye(3)), axis=1)
two = np.concatenate((K_global/-((rho * A * l) / 2),C_global/-((rho * A * l) / 2)), axis=1)
Ac = np.concatenate((one,two))

fs = 3000         # Sampling Frequency 
T = 1 / fs      # Sampling period

t = np.arange(0, 100, T)

# Compute the matrix exponential A
A = la.expm(Ac * T)
C = Ac[3:, :]

Y = np.zeros((C.shape[0], len(t)))

np.random.seed(1)
v = 1e-2 * np.random.randn(A.shape[0], len(t))
w = 1e-3 * np.random.randn(C.shape[0], len(t))
x = np.zeros(A.shape[0])

# Simulate the system iteratively
for i in range(len(t)):
    Y[:, i] = np.dot(C, x) + w[:, i]   # Acceleration responses into a matrix by rows
    x = np.dot(A, x) + v[:, i]
    
(freq, SSY)= welch(Y, fs, nperseg=1024)

## Calculate modal parametrs

fn, zeta, Phi = modalparams(A, C, T)

#List fn, zeta

fnl = fn[0].tolist()
zetal = zeta[0].tolist()

# Create an array for the x-axis
modes = np.arange(1, len(fnl) + 1)

# Plot the time history
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, Y.T)
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (ms^-2)')
plt.axis('tight')
#plt.xlim(0,100)
plt.grid(True)
plt.subplot(2, 1, 2)
for i in range(SSY.shape[0]):
    plt.plot(freq, 10 * np.log10(SSY[i]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB)')
plt.axis('tight')
plt.grid(True)

yl = plt.ylim()
for fn_val in fn[0]:  # Assuming fn is a 2D numpy array, use fn[0] to access the first row
    plt.axvline(x=fn_val, color='k', linestyle='--')

plt.show()

# Plot natural frequencies
fig = plt.figure(figsize=(10, 3), dpi=80)

plt.subplot(121)
plt.plot(modes, fnl,  'o', color='black')
plt.xlabel('Mode')
plt.ylabel('Frequency (Hz)')
plt.xticks(np.arange(1, len(fnl) + 1))
plt.title('Natural frequencies')
plt.grid(True)

# Plot damping ratios
plt.subplot(122)
plt.plot(modes, zetal,  'o', color='black')
plt.xlabel('Mode')
plt.ylabel('Damping ratio')
plt.xticks(np.arange(1, len(zetal) + 1))
plt.title('Damping ratios')
plt.grid(True)

plt.show
