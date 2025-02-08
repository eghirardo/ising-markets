import numpy as np
import matplotlib.pyplot as plt
import random
from ipywidgets import interact, IntSlider
import scipy.linalg as la

def lattice_connection_matrix(side: int, dim: int) -> np.ndarray:
    """
    Generates a lattice connection matrix for a given side length and dimension.

    Parameters:
    side (int): The length of one side of the lattice.
    dim (int): The dimension of the lattice.

    Returns:
    numpy.ndarray: The connection matrix representing the lattice.

    The function constructs a circulant matrix for the given side length and 
    then uses the Kronecker product to extend this matrix to higher dimensions.
    For 1D, it returns the circulant matrix directly. For 2D, it constructs the 
    connection matrix by combining Kronecker products of the circulant matrix 
    and the identity matrix. For higher dimensions, it recursively constructs 
    the connection matrix by adding the Kronecker product of the identity matrix 
    and the path matrix.
    """
    path = la.circulant([0,1] + [0]*(side-3) + [1])
    I = np.eye(side)
    if dim == 1:
        return path
    elif dim == 2:
        return np.kron(path,I) + np.kron(I,path)
    else:
        return lattice_connection_matrix(side, dim-1) + np.kron(I, path)

class SpinMarketModel:
    """
    A class to represent a spin market model using the Bornholdt model.
    Attributes
    ----------
    size : int
        The number of spins in the system (default is 1024).
    J : float
        Ferromagnetic coupling constant (default is 1).
    alpha : float
        Global anti-ferromagnetic coupling constant (default is 4).
    T : float
        Temperature of the system (default is 1.5).
    steps : int
        Number of Monte Carlo sweeps (default is 10000).
    spins : numpy.ndarray
        Array representing the spin states of the system.
    local_field_func : function
        Function to calculate the local field at a given spin.
    connection_matrix : numpy.ndarray
        Matrix representing the connections between spins.
    Methods
    -------
    metropolis_step():
        Perform a single Metropolis update.
    run_simulation(verbose=False):
        Run the Monte Carlo simulation.
    plot_magnetization(spin_series):
        Plot the time series of magnetization.
    """
    def __init__(self,
                 size: int = 1024,
                 J: float = 1,
                 alpha: float = 4,
                 T: float = 1.5,
                 steps: int = 10000,
                 local_field_func: callable = None,
                 connection_matrix: np.ndarray = None):
        self.size = size
        self.J = J  # Ferromagnetic coupling
        self.alpha = alpha  # Global anti-ferromagnetic coupling
        self.T = T  # Temperature
        self.steps = steps  # Number of Monte Carlo sweeps
        self.spins = np.random.choice([-1, 1], size)  # Initialize spins randomly
        self.local_field_func = local_field_func
        self.connection_matrix = connection_matrix if connection_matrix is not None else np.ones((size, size)) - np.eye(size)
    
    def metropolis_step(self):
        """
        Perform a single Metropolis update on the spin configuration using heat bath dynamics.

        Steps:
        1. Randomly select a spin from the configuration.
        2. Calculate the probability of the spin being +1 using the local field
           and the temperature.
        3. Flip the spin to +1 with the calculated probability, otherwise set it to -1.

        Note:
        - The Boltzmann constant is neglected as it has no physical meaning in this application.

        Attributes:
        - self.size: The number of spins in the system.
        - self.local_field_func: A function that calculates the local field at a given spin.
        - self.T: The temperature of the system.
        - self.spins: The array representing the spin configuration.
        """
        """Perform a single Metropolis update"""
        for _ in range(self.size):
            i = np.random.randint(0, self.size)
            # here we neglect the boltzmann constant as it has no physical meaning in our application
            p = 1/(1 + np.exp(-2 * self.local_field_func(self, i) / self.T))
            if np.random.rand() < p:
                self.spins[i] = +1
            else:
                self.spins[i] = -1
    
    def run_simulation(self, verbose=False):
        """Run the Monte Carlo simulation"""
        spin_series = []
        if verbose:
            print("Running simulation with {} spins, J={}, alpha={}, T={}, steps={}".format(self.size, self.J, self.alpha, self.T, self.steps))
        for step in range(self.steps):
            if verbose and step % int(self.steps/10) == 0:
                print(f"Step {step}/{self.steps}")
            self.metropolis_step()
            spin_series.append(np.copy(self.spins))
        print('Simulation finished')
        return spin_series

    def plot_magnetization(self, spin_series):
        """Plot the time series of magnetization"""
        magnetization_series = [np.sum(spin) for spin in spin_series]
        plt.figure(figsize=(10, 5))
        plt.plot(magnetization_series, label="Magnetization")
        plt.xlabel("Monte Carlo Steps")
        plt.ylabel("Magnetization")
        plt.title("Magnetization over Time")
        plt.legend()
        plt.show()
    
    
# need to update this class to use the general connection matrix function !!!
class LatticeSpinMarketModel(SpinMarketModel):
    def __init__(self, side=32, dim=2, J=1, alpha=4, T=1.5, steps=10000, local_field_func=None):
        self.dim = dim
        self.size = side**self.dim
        self.J = J  # Ferromagnetic coupling
        self.alpha = alpha  # Global anti-ferromagnetic coupling
        self.T = T  # Temperature
        self.steps = steps  # Number of Monte Carlo sweeps
        self.spins = np.random.choice([-1, 1], (self.size))  # Initialize spins randomly
        self.local_field_func = local_field_func
        self.connection_matrix = lattice_connection_matrix(side=side, dim=self.dim)

    def plot_lattice(self, spin_series, t=None, interactive=False):
        """
        Plots the 2D lattice at time t.
        """
        if self.dim != 2:
            raise ValueError("This method is only implemented for 2D lattices")
        if t is None:
            t = len(spin_series) - 1
        if interactive:
            interact(lambda t: self.plot_lattice(spin_series, t), t=IntSlider(min=0, max=len(spin_series)-1, step=1, value=0))
        else:
            side = int(np.sqrt(self.size))
            plt.figure(figsize=(8, 8))
            plt.imshow(spin_series[t].reshape(side, side), cmap='binary', interpolation='none')
            plt.title(f'Lattice at time t={t}')
            plt.colorbar(label='Value')
            plt.show()