from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import eigvals, eigh, solve

# Structural properties
E = 200e9 # Young's modulus
rho = 7.8e3 # Density
L = 5 # Length
d = 30e-3 # Diameter

A = np.pi / 4 * d ** 2 # Cross-sectional area
I = ( np.pi * d ** 4 ) / 64 # Moment of inertia

# Element configuration
n = 3 # Number of dofs (measuring points with accelerometers)
l = L / (n + 1) # Length of each element

ko = E * A / l * np.array([[1, -1], [-1, 1]]) # Stiffness matrix (element)
m1_2 = (rho * A * l) / 2 * np.eye(2) # Mass of each element
m3 = (rho * A * l) * np.eye(2)


# Element mass and stiffness matrices
Ko = np.zeros((n + 2, n + 2))
Mo = np.zeros((n + 2, n + 2))

for i in range(1, n + 2):
  ind = np.arange(i - 1, i + 1)
  Ko[ind[:, np.newaxis], ind] += ko
  if i <= n:
    Mo[ind[:, np.newaxis], ind] += m1_2
  else:
    Mo[ind[:, np.newaxis], ind] += m3

Ko = Ko[1:n + 1, 1:n + 1]
Mo = Mo[1:n + 1, 1:n + 1]
print()
print("Stiffness Matrix, K:", Ko, '\n')
print("Mass Matrix, M:", Mo, '\n')
print()

# print("\nKo", Ko)
# print("\nMo", Mo)

# exit(0)

alpha = 1e-5
beta = 1e-2
Co = alpha * Ko + beta * Mo

# print("\nM^-1Ko", -np.linalg.inv(Mo)*Ko)
eigvals, eigvecs = eigh(Ko, Mo)

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

print("\nEigenvalues", eigvals)
print("\nEigenvecs", eigvecs)

print("\nNatural Frequencies:")
for i, val in enumerate(eigvals):
  print(f"\t{i+1}: {np.sqrt(val.real) / (2 * np.pi):.2f} Hz")

print("\nMode Shapes:")
mode_shapes = list(map(normalize_vector, eigvecs))
for i in range(len(mode_shapes)):
   print(i + 1, mode_shapes[i])

# Force input matrix
d = np.zeros(n)
d[-1] = 1

# Extract natural frequencies
omega0 = np.sqrt(eigvals.real)

# Extract column mode shapes
psi = eigvecs

# Damping ratios of system
zeta = np.diag(psi.T.dot(Co).dot(psi)) / (2 * omega0)

plt.figure(figsize=(5, 5))
plt.plot(np.arange(1, n + 1), omega0 / (2 * np.pi), 'o', color='black')
plt.grid(True)
plt.xlabel('Mode')
plt.ylabel('Frequency (Hz)')
plt.xlim(0.5, n + 0.5)
plt.xticks(np.arange(1, n + 1))
plt.title('Natural frequencies')
# plt.show()

# Create a figure with specified position
fig = plt.figure(figsize=(9, 9), dpi=80)  # Adjust the figsize and dpi as needed

# Loop for subplots
for i in range(1, 4):
    ax = fig.add_subplot(2, 2, i)
    x_values = np.concatenate(([0], np.arange(1, n + 1) * l))
    y_values = np.concatenate(([0], psi[:, i - 1]))
    
    ax.plot(x_values, y_values, '-o', color='black', markeredgecolor='black', markerfacecolor='white')
    ax.grid(True)
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Displacement (m)')
    ax.set_title(f"Mode {i}\nFrequency {omega0[i - 1] / (2 * np.pi):.2f} (Hz)\nDamping {100 * zeta[i - 1]:.2f}%")

# Adjust subplot layout
fig.tight_layout()

# Show the plot
plt.show()


### FRF Matrix ###
nf = 500 # number of sampling frequency point for FRF calculation
fs = 4000 # sampling frequency
f = np.linspace(0, fs/2, nf)     # Generating the Frequency vector (Hz) 
Om = 2 * np.pi * f               # Frequency vector (rad/s)

Ho = np.zeros((n, nf), dtype=complex) # Generating matrix space

for i in range(nf):
  A = Ko - Mo * Om[i]**2 + 1j * Co * Om[i]
  Ho[:, i] = solve(A, d)

plt.figure(figsize=(10, 25))
for i in range(n):
    plt.subplot(n, 1, i + 1)
    plt.semilogy(f, np.abs(Ho[i, :]), color='black')
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FRF magnitude (m)')
    plt.title(f'FRF DOF {i + 1}')
    plt.ylim(1e-10, 1e-5)
    plt.xlim(0, fs/2)

plt.show()




# exit(0)

# Construct system matrix
# A = np.block([[np.zeros_like(Ko), np.eye(n)], [-np.linalg.inv(Mo)*Ko, -np.linalg.inv(Mo)*Co]])