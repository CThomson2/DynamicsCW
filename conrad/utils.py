"""
OMA: Stochastic System Identification for Python
    
Author: Dashty Samal Rashid
Support email: D.S.Rashid@ed.ac.uk

The Univeristy of Edinburgh, IIE

Version: 1.1

"""

#=============================================================================

import numpy                   as np
import matplotlib.pyplot       as plt
from scipy.linalg              import sqrtm, pinv, eigvals
from scipy.linalg              import svd                         
from scipy.signal              import welch
import warnings
import scipy.linalg as la

#=============================================================================



def modalparams(A, C, dt):
    """
    Calculate modal parameters (natural frequencies, damping ratios, and mode shapes) for a dynamic system.

    Parameters:
    - A (list or matrix): List of system matrices or a single system matrix.
    - C (list or matrix): List of output matrices or a single output matrix.
    - dt (float): Time step or time increment used for the analysis.

    Returns:
    - f (list of arrays): List of natural frequencies (in Hertz) for each system.
    - zeta (list of arrays): List of damping ratios for each system.
    - Phi (list of arrays): List of mode shapes for each system.
    
    """
    # Ensure A and C are lists, even if they are initially single matrices
    if not isinstance(A, list):
        A = [A]
    if not isinstance(C, list):
        C = [C]
    
    # Initialize empty lists to store modal parameters
    f = [None] * len(A)  # List for natural frequencies
    zeta = [None] * len(A)  # List for damping ratios
    Phi = [None] * len(A)  # List for mode shapes

    # Loop through each matrix in the input list A 
    for i in range(0, len(A)):
        
        # Calculate eigenvalues (d) and eigenvectors (v) of the matrix A[i]
        d, v = np.linalg.eig(np.array(A[i]))
        
        # Compute natural frequencies (in Hertz) from eigenvalues
        lam = np.log(d) / dt
        f[i] = np.abs(lam) / (2 * np.pi)
        
        # Sort frequencies and corresponding damping ratios
        f[i], I = np.sort(f[i]), np.argsort(f[i])
        zeta[i] = -np.real(lam) / np.abs(lam)
        zeta[i] = zeta[i][I]
        
        # Calculate mode shapes by multiplying matrix C[i] with eigenvectors v
        Phi[i] = C[i] @ v
        Phi[i] = Phi[i][:, I]
        
        # Ensure frequencies are unique and update corresponding parameters
        f[i], I = np.unique(f[i], return_index=True)
        zeta[i] = zeta[i][I]
        Phi[i] = Phi[i][:, I]
    

    
    # Return the computed modal parameters for all matrices
    return f, zeta, Phi

#=============================================================================

def plotBuildingModes(Phi):
    """
    Plot the mode shapes of a building structure.

    Parameters:
    - Phi (array): Matrix of mode shapes with shape (number_of_floors, number_of_modes).

    This function visualizes the mode shapes of a building structure. It plots each mode shape
    as a set of displacements along the building's floors.

    Each column in the input matrix Phi represents a mode shape, and each row represents the
    displacement of each floor at a particular mode. The function plots the mode shapes for
    all available modes in separate subplots.

    Returns:
    - None
    
    """

    plt.clf()
    
    for i in range(Phi.shape[1]):
        # Create a subplot for each mode shape
        plt.subplot(1, Phi.shape[1], i+1)
        
        z = Phi[:, i]
        
        # Compute the phase of the mode shape vector
        phase = np.column_stack((np.cos(np.angle(z)), np.sin(np.angle(z))))
        
        # Choose the mode shape phase that maximizes the displacement
        idx = np.argmax([np.sum(np.abs(phase[:, 0])), np.sum(np.abs(phase[:, 1]))])
        shape = np.abs(z) * phase[:, idx]
        
        # Plot the mode shape
        plt.plot([0] + shape.tolist(), list(range(Phi.shape[0] + 1)), '.-', markersize=15, color='k')
        
        # Plot the 'other side' of the mode shape
        #plt.plot([0] + (-shape).tolist(), list(range(Phi.shape[0] + 1)), '.-', markersize=15)
        
        # Plot the building centerline
        plt.axvline(x=0, ymin=0, ymax=Phi.shape[0], color='k', linestyle='--')
        
        plt.yticks(list(range(1, Phi.shape[0] + 1)))
        plt.xticks([])

        if i == 0:
            plt.ylabel('Floor No.')  # Only label the y-axis at the first plot
        else:
            plt.yticks([])

        plt.xlabel('Mode ' + str(i + 1))
        plt.axis('tight')

    # Adjust subplot layout to remove extra white space
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1)

    plt.show()

#=============================================================================

def ssicov(data, order, fs, s=None, ts:float = 5.):
    """
    Covariance-based stochastic subspace identification (SSI-cov).

    Parameters:
    - Y: Sensor data matrix
    - order: Desired maximum model order to identify (scalar)
    - s: Number of block rows in the block Hankel matrix; should be at least ceil(order/ns)
         to obtain results up to the desired model order. Generally, s > order.
    - fs : float Sampling frequency [Hz] of the `data` time series
    - ts: float, optional time lag [s] for covariance calculation. Defaults to 5. s

    Returns:
    - A: List of state transition matrices for model orders {i}
    - C: List of output matrices for model orders {i}
    - G: List of next state output covariance matrices for model orders {i}
    - R0: Zero-lag output covariances
    """
    
    # Calucate the time step of the data
    n_ch = data.shape[0]
    dt = 1 / fs
    
    # Input conditioning, assumes more samples than sensors
    data = data.T if data.shape[0] > data.shape[1] else data
    ns, nt = data.shape  # ns = # of sensors, nt = # of samples
    
    # Set the number of block rows in the block Hankel matrix
    if s is None or s < np.ceil(order / ns):
        s = int(np.ceil(order / ns))  # s is at least ceil(order/ns), better results obtained using more time lags
        print('Warning: Block Hankel matrix size too small, using s =', s)
        
    R0 = 1 / nt * np.dot(data[:, :nt], data[:, :nt].T)
    
    # get impulse response function via Next method
    irf = next_method(data, dt, ts)
    
    # Block Hankel matrix of output covariances
    print('Forming block Hankel matrix...')
    
    # get hankel matrix
    U, S, V = blockhankel(irf)

    # truncate at max model order
    S = S[:order]
    U = U[:, :order]
    V = V[:, :order]
    
    # Output cell arrays
    A = [None] * order
    C = [None] * order
    G = [None] * order

    # Loop over model orders
    for i in range(0, order):
        U1 = U[:, :i]  # truncate decomposition at model order
        V1 = V[:, :i]  # truncate decomposition at model order
        ss = np.diag(np.sqrt(S[:i]))  # square root of truncated singular values
        Obs = U1 @ ss  # observability matrix
        Con = ss @ V1.T  # controllability matrix (reversed if using Toeplitz)
        A[i] = np.linalg.pinv(Obs[:-ns, :]) @ Obs[ns:, :]  # system matrix
        C[i] = Obs[:ns, :]  # output matrix
        G[i] = Con[:, :ns]  # next state output covariance matrix
        # G[i] = Con[:, -ns:]  # if using Toeplitz matrix
        
    print('SSI-cov finished.')
    return A, C, G, R0

#=============================================================================

def welch_psd(data, fs, nperseg=1024):
    """
    Calculate Welch Power Spectral Density (PSD).

    Parameters:
    - Y: matrix containing output samples
    - fs: sampling frequency [Hz] of the `Y` time series
    - nperseg: number of samples per segment, defaults to 1024

    Returns:
    - f_x: array of sample frequencies
    - pxx: power spectral density
    """
    f_x, pxx = welch(data, fs, nperseg=nperseg)
    pxx = pxx.mean(axis=0)

    return f_x, pxx

#=============================================================================

def plot_stabilization_diagram(A, C, data, fs, dt, order, win=None, err=None):
    """
    Creates a stabilization diagram for modal analysis purposes.

    Parameters:
    - A: list of system matrices
    - C: list of output matrices
    - data: test data used for model identification
    - dt: sampling period of output data y
    - win: optional window to be used for estimation of output power spectrums
    - err: 3-element vector of percent errors for stability criteria
           (frequency, damping, and modal assurance criterion),
           default is [0.01, 0.05, 0.98]

    Returns:
    - IDs: list containing logical vectors of indices of stable
           modes on the diagram for all model orders of the identified system
    """

    # Default error values if not specified
    if err is None:
        print('No stabilization criteria specified, using default settings for stabilization criteria')
        err = [0.01, 0.05, 0.98]

    # Check for out-of-bounds error vector
    if any(np.array(err) < 0) or any(np.array(err) > 1):
        raise ValueError('Vector err out of bounds in the range 0 to 1')

    # Generate the complex mode indicator function (CMIF)
    f_x, pxx = welch_psd(data, fs, nperseg=1024)

    # Create subplots with shared x-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Plot the first singular value of the CMIF to aid in pole selection
    a = ax1.semilogy(f_x, pxx, c='tab:blue', label='PSD')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('PSD', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Input conditioning, samples go down rows (assumes more samples than channels)
    data = np.atleast_2d(data)
    if data.shape[0] < data.shape[1]:
        data = data.T

    # Generate modal decompositions
    f, psi, Phi = modalparams(A, C, dt)

    # Loop over model orders
    IDs = []
    for i in range(len(A) - 1):
        
        # Round the complex numbers to a specific number of decimal places
        rounded_f1 = np.round(f[i], decimals=10)
        rounded_f2 = np.round(f[i+1], decimals=10)
        
        # Unique for the current array
        f1, I1 = np.unique(rounded_f1, return_index=True)

        # Unique for the next array
        f2, I2 = np.unique(rounded_f2, return_index=True)

        psi1 = psi[i][I1]
        psi2 = psi[i + 1][I2]
        phi1 = Phi[i][:, I1]
        phi2 = Phi[i + 1][:, I2]

        # Frequency stability criteria
        ef = np.sum(np.abs(np.subtract.outer(f1, f2) / np.expand_dims(f1, axis=1)) <= err[0], axis=1)

        # Damping stability criteria
        epsi = np.sum(np.abs(np.subtract.outer(psi1, psi2) / np.expand_dims(psi1, axis=1)) <= err[1], axis=1)

        # MAC stability criteria
        mac_vals = np.zeros(len(f2))
        ephi = np.zeros(len(f1))

        # Check each mode shape vector with every other mode shape vector from the next model order up
        for j in range(len(f1)):
            for k in range(len(f2)):
                mac_vals[k] = Mac(phi1[:, j], phi2[:, k])
            ephi[j] = np.sum(mac_vals >= err[2]).astype(bool)


        # Valid (stable) poles
        IDs.append(I1[np.logical_and(np.logical_and(ef, epsi), ephi)])
        
        # Plot unstable poles
        unstable = f1[~(np.logical_and(np.logical_and(ef, epsi), ephi))]
       
        if len(unstable) > 0:
            b = ax2.plot(unstable, [i] * len(unstable), 'ro', label='Unstable')

        # Plot stable poles
        stable = f1[np.logical_and(np.logical_and(ef, epsi), ephi)]
      
        if len(stable) > 0:
            c = ax2.plot(stable, [i] * len(stable), 'gx', label='Stable')

    # Label axes
    ax2.set_ylabel('Model Order', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # added these three lines
    lns = a+b+c
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    # Set limits on x-axis
    ax1.set_xlim([f_x[0], f_x[-1]])
    ax2.set_ylim([0, order])
    ax2.set_yticks(np.arange(0, order+1, 1))
    
    # Show the plot
    plt.title('Stabilization Diagram')
    plt.show()

    return IDs


#=============================================================================

def Mac(phi1,phi2):
    '''
    This function returns the Modal Assurance Criterion (MAC) for two mode 
    shape vectors.
    
    If the input arrays are in the form (n,) (1D arrays) the output is a 
    scalar, if the input are in the form (n,m) the output is a (m,m) matrix
    (MAC matrix).
    
    ----------
    Parameters
    ----------
    Fi1 : array (1D or 2D)
        First mode shape vector (or matrix).
    Fi2 : array (1D or 2D)
        Second mode shape vector (or matrix). 
        
    -------
    Returns
    -------
    MAC : float or (2D array)
        Modal Assurance Criterion.
    '''
    
    MAC = np.abs(phi1.conj().T @ phi2)**2 / \
        ((phi1.conj().T @ phi1)*(phi2.conj().T @ phi2))
    warnings.simplefilter('ignore')
        
    return MAC


#=============================================================================


def blockhankel(irf):
    """
    Generate block Hankel matrix using covariances from output data.

    Parameters:
    - Y_ref: Reference data
    - Y_all: Moving sensor data; if not using reference-based identification, then Y_all = Y_ref
    - s: Number of time lags used in covariance calculation

    Returns:
    - H: Block Hankel matrix
    """

    n_len = round(irf.shape[2] / 2) - 1
    n_ch = irf.shape[0]
    hank = np.zeros((n_ch * n_len, n_ch * n_len), dtype=np.complex128)

    for i in range(n_len):
        for j in range(n_len):
            hank[i * n_ch: (i + 1) * n_ch, j * n_ch: (j + 1)
                 * n_ch] = irf[:, :, n_len + i - j]
    
    # balanced realization (no weighting matrices)
    print('Performing singular value decomposition...')
  
    U, S, V = np.linalg.svd(hank)
    
    return U, S, V

#=============================================================================

def next_method(data, dt, ts):
    """
    Computes the Impulse Response Function (IRF) using the Fast Fourier Transform (FFT) method.

    Parameters:
    - data: 2D array, shape (n_ch, n_samples)
        Input data representing signals from multiple channels.
    - dt: float
        Sampling period of the input data.
    - ts: float
        Time span for which the IRF is calculated.

    Returns:
    - irf: 3D array, shape (n_ch, n_ch, m)
        Array containing complex-valued IRFs for each pair of channels.
        'm' is determined based on the given time span and sampling period.
    """

    # FFT-based Impulse Response Function (IRF) calculation

    # Number of channels in the data
    n_ch = data.shape[0]

    # Number of samples to consider for IRF
    m = round(ts / dt)

    # Initialize an array to store the complex-valued IRFs
    irf = np.zeros((n_ch, n_ch, m), dtype=np.complex128)

    # Iterate over each pair of channels
    for i in range(n_ch):
        for j in range(n_ch):
            # Compute the FFT of the input signals for both channels
            y1 = np.fft.fft(data[i])
            y2 = np.fft.fft(data[j])

            # Compute the inverse FFT of the product of the FFTs
            h = np.fft.ifft(y1 * np.conj(y2))

            # Store the computed IRF, keeping only the first 'm' samples
            irf[i, j, :] = h[:m]

    return irf

