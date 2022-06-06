import numpy as np
import json
from scipy.interpolate import interp1d
from numpy.linalg import eig

from dataclasses import dataclass
from typing import Optional, List, Tuple
from numpy.typing import ArrayLike

# Model structure

@dataclass
class Process:
    """object to hold individual process information."""

    name: str
    timescale: float
    compartment_from: str
    compartment_to: Optional[str] = None
    reference: Optional[str] = ''

    def get_k(self, t: float = 0) -> float:
        """First order rate of process."""
        return 1/self.timescale


class Input(object):
    """object to hold individual input information."""

    name: str
    raw_E: ArrayLike
    compartment_to: str
    raw_t: Optional[ArrayLike] = None
    reference: Optional[str] = ''

    def __init__(self, name, E, t, cto, ref=''):
        self.name = name
        self.raw_E = E
        self.raw_t = t
        self.compartment_to = cto
        self.reference = ref

        if t is None:
            self.E_of_t = lambda x: self.raw_E
        else:
            pad_t = np.concatenate(([-4e12], self.raw_t, [4e12]))
            pad_E = np.concatenate((self.raw_E[0:1], self.raw_E,
                                    self.raw_E[-1:]))
            self.E_of_t = interp1d(pad_t, pad_E)

    def get_input(self, t: float = 0) -> float:
        """Get input rate from this source at given time."""
        return self.E_of_t(t)

    def get_raw(self) -> Tuple[ArrayLike, ArrayLike]:
        """Get raw input values."""
        return self.raw_E, self.raw_t


class Model(object):
    """Box model implementation.
    Attributes:
    compartments: names of compartments (list of strings)
    compartment_indices: look-up for compartment index (dict)
    N: number of compartments (int)
    processes: collection of process objects (dict)
    inputs: collection of Input objects (dict)
    matrix: transfer matrix representation of processes
    inputs_to_use: which inputs to use, by name ('all' or list of strings)
    Methods:
    add_process: add new process to collection
    add_input: add new Input to collection
    build_matrix: translate process collection to matrix form
    get_inputs: look up total inputs at a given time
    run: solve for compartment masses for given times
    """

    def __init__(self, compartments):
        """Initialize model based on listed compartments."""

        self.compartments = compartments
        self.compartment_indices = {
            c: i for i, c in enumerate(compartments)}
        self.N = len(compartments)
        self.processes = {}
        self.inputs = {}
        self.matrix = None
        self.inputs_to_use = 'all'

    def set_inputs_to_use(self, input_list):
        """Set whitelist of inputs."""
        self.inputs_to_use = input_list


    def add_process(self, process):
        """Add a process to the model's collection.
        Parameters:
        process: individual process info as process object
        """

        self.processes[process.name] = process

    def add_input(self, inp):
        """Add an input to the model's collection.
        Parameters:
        inp: individual input info as Input object
        """

        self.inputs[inp.name] = inp

    def build_matrix(self, ):
        """Translate collection of processes to matrix form."""

        self.matrix = np.zeros((self.N, self.N))
        for name, process in self.processes.items():
            i = self.compartment_indices[process.compartment_from]
            # check if flow stays in system:
            if process.compartment_to is not None:
                j = self.compartment_indices[process.compartment_to]
                self.matrix[j, i] += process.get_k()
            self.matrix[i, i] -= process.get_k()

    def get_inputs(self, t=0):
        """Get total inputs at given time.
        Parameters:
        t: time (float) [optional]
        Returns: array of total emissions
        """

        if self.inputs_to_use == 'all':
            input_list = self.inputs.keys()
        else:
            input_list = self.inputs_to_use

        E = np.zeros(self.N)
        for input_name in input_list:
            inp = self.inputs[input_name]
            i = self.compartment_indices[inp.compartment_to]
            E[i] += inp.get_input(t)

        return E

    def decompose(self):
        """ Decompose Matrix into eigenvalues/vectors.
        Returns:
            eigenvalues (array): unordered array of eigenvalues
            eigenvectors (array): normalized right eigenvectors
                   ( eigenvectors[:,i] corresponds to eigenvalues[i] )
        """
        eigenvalues, eigenvectors = eig(self.matrix)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.inverse_eigenvectors = np.linalg.inv(self.eigenvectors)
        self.timescales = -1./np.float64(np.real(eigenvalues))
        self.residence_times = -1/np.float64(np.diagonal(self.matrix))
        return eigenvalues, eigenvectors

    def homogeneous_timestep(self, dt, initial_conditions):
        """Calculate result of unforced evolution over dt.
        initial_conditions in eigenspace
        """
        eigIC = initial_conditions
        in_eig_space = np.exp(self.eigenvalues * dt) * eigIC
        return in_eig_space

    def inhomogeneous_timestep(self, dt, eigf0, eigf1=None):
        """Calculate result of forcing that changes linearly over dt.
        If eigf1 not given, then constant eigf0 forcing used.
        """

        b = (eigf1-eigf0)/dt
        c = eigf0
        ev = self.eigenvalues
        in_eig_space = (-ev*b*dt - ev*c - b + (ev*c+b)*np.exp(ev*dt))/(ev**2)
        return in_eig_space

    def run(self, times, initial_conditions=None):
        if initial_conditions is None:
            initial_conditions = np.zeros(self.N)

        self.decompose()
        all_times = list(times)
        for input_name in self.inputs:
            thist = self.inputs[input_name].raw_t
            if thist is not None:
                all_times = np.concatenate((all_times, thist))
        needed_times = np.sort(np.unique(all_times))
        needed_times = needed_times[needed_times >= times[0]]
        needed_times = needed_times[needed_times <= times[-1]]
        solution = np.zeros((self.N, len(needed_times)))
        output = np.zeros((self.N, len(times)))
        solution[:, 0] = np.dot(self.inverse_eigenvectors, initial_conditions)
        output[:, 0] = initial_conditions
        j = 1
        for i in range(1, len(needed_times)):
            dt = needed_times[i]-needed_times[i-1]
            Mh = self.homogeneous_timestep(dt, solution[:, i-1])
            Mi = self.inhomogeneous_timestep(dt,
                                             np.dot(self.inverse_eigenvectors, self.get_inputs(
                                                 needed_times[i-1])),
                                             np.dot(self.inverse_eigenvectors, self.get_inputs(needed_times[i])))
            solution[:, i] = Mh + Mi
            if needed_times[i] in times:
                output[:, j] = np.real(np.dot(self.eigenvectors, Mh + Mi))
                j += 1

        return output.T, times

    def run_euler(self, tstart, tend, dt=0.01, initial_conditions=None):
        """Calculate masses in all compartments through time.
        Parameters:
        tstart: start time (float)
        tend: end time (float)
        dt: time step (float) [optional]
        initial_conditions: initial mass in each box
                         (None or array) [optional]
        Returns: mass, time
        mass: mass in each compartment through time
                 (2D array of size (Ntime, Ncompartmets))
        time: time axis of solution (1D array of length Ntime)
        """

        nsteps = int((tend - tstart) / dt) + 1
        time = np.linspace(tstart, tend, nsteps)
        M = np.zeros((nsteps, self.N))
        if initial_conditions is None:
            M0 = np.zeros(self.N)
        else:
            M0 = initial_conditions
        M[0, :] = M0
        for i, t in enumerate(time[:-1]):
            dMdt = np.dot(self.matrix, M[i, :]) + self.get_inputs(t)
            dM = dMdt * dt
            M[i+1, :] = M[i, :] + dM

        return M, time

    def get_steady_state(self, input_vector):
        """Calculate the steady state reservoirs associated with given inputs.
        model: Model object
        input_vector: vector of inputs to each compartment
        returns: steady state reservoirs
        """
        input_vector = np.array(input_vector)
        return np.linalg.solve(self.matrix, -input_vector)

    def as_json(self):
        """Get JSON represenation of model.
        Will return a JSON representation containing Processes, Emissions, Compartments.
        This representation can be loaded for future use or with other back-ends like boxey.
        """
        to_json = {'Compartments': self.compartments,
                   'Processes': [], 'Inputs': []}

        for n, process in sorted(self.processes.items()):
            to_json['Processes'].append({'Name': process.name, 'Timescale': process.timescale,
                                         'Compartment_to': process.compartment_to,
                                         'Compartment_from': process.compartment_from
                                         })
        for n, inp in sorted(self.inputs.items()):
            to_json['Inputs'].append({'Name': inp.name,
                                      'Compartment_to': inp.compartment_to,
                                      'Magnitude': inp.raw_E,
                                      'Time': inp.raw_t})
        return json.dumps(to_json, indent=4, sort_keys=True)

    def to_json(self, filename):
        with open(filename, 'w') as f:
            f.write(self.as_json())

# Helper functions


def model_from_json(filename):
    """Construct model from JSON definition file"""
    return


def create_model(compartments, processes):
    """Turn list of compartment names and list of processes into model.
    compartments: list of compartment names
    processes: list of process objects
    returns: created Model object
    """
    model = Model(compartments)

    for process in processes:
        model.add_process(process)
    model.build_matrix()  # model creates matrix

    return model


def add_inputs(model, input_list):
    """Add list of Inputs to model.
    model: Model object
    input_list: list of Input objects
    returns: updated Model object
    """

    for inp in input_list:
        model.add_input(inp)

    return model
