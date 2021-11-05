# Quantum_Dot.py
#
# Class container for 1D quantum dot problems
# Currently supports:
#   - Time-independent solution using an arbitrary potential
#   - Time-dependent system evolution again using an arbitrary (time-dependent or not) potential
#
# Hugo Sebesta, 2021


# Relevant imports
import numpy as np
from numpy import linalg as LA

# Quantum dot class
class Quantum_Dot:
    'Class for a quantum dot problem. Contains methods: \
    - __init__() \
    - setter() \
    - generate_hamiltonian() \
    - populate_hamiltonian() \
    - solve_eigenproblem() \
    - eigen_energies() \
    - eigen_wavefunctions() \
    - evolution()'

    # 1 Dimensional (dot)
    dim = 1

    def __init__(self, n, h_bar = 1, m = 1, V_func = None, debug = False):
        # Initialisation
        self.n = n
        self.h_bar = h_bar
        self.m = m
        self.V_func = V_func

        self.debug = debug

        self.Hamiltonian = None

        self.E = None
        self.Psi = None

        # Evolution
        self.Psi_vs_t = None
        self.E_vs_t = None

        self.t_0 = None
        self.t_max = None
        self.delta_t = None
        self.evolution_pre_func = None
        self.evolution_post_func = None
        self.step_pre_func = None
        self.step_post_func = None
        self.pre_evol_variables = None

    def setter(self, h_bar = None, m = None, V_func = None, hamiltonian = None, Eigenenergies = None, Psis = None, t_0 = None, t_max = None, delta_t = None, evolution_pre_func = None, evolution_post_func = None, step_pre_func = None, step_post_func = None, pre_evol_variables = None):
        'Allows the user to efficiently set values to the class'
        if h_bar != None:
            self.h_bar = h_bar
        if m != None:
            self.m = m
        if V_func != None:
            self.V_func = V_func
        if hamiltonian != None:
            if hamiltonian.shape() != (self.n, self.n):
                print("Dimension mismatch. Did not continue")
                return

            self.Hamiltonian = hamiltonian
        if Eigenenergies != None:
            if Eigenenergies.size() != self.n:
                print("Dimension mismatch. Did not continue")
                return

            self.E = Eigenenergies
        if Psis != None:
            if Psis.size() != self.n:
                print("Dimension mismatch. Did not continue")
                return

            self.Psi = Psis
        if t_0 != None:
            self.t_0 = t_0
        if t_max != None:
            self.t_max = t_max
        if delta_t != None:
            self.delta_t = delta_t
        if evolution_pre_func != None:
            self.evolution_pre_func = evolution_pre_func
        if evolution_post_func != None:
            self.evolution_post_func = evolution_post_func
        if step_pre_func != None:
            self.step_pre_func = step_pre_func
        if step_post_func != None:
            self.step_post_func = step_post_func
        if pre_evol_variables != None:
            self.pre_evol_variables = pre_evol_variables

    def mode_debug(self):
        self.debug = True

    def mode_run(self):
        self.debug = False

    def generate_hamiltonian(self, h_bar = None, m = None, no = None, V_func = None, **V_params):
        'Generates a hamiltonian in the most general case. Note that params should NOT include x'
        # Loading parameters
        if h_bar == None:
            _h_bar = self.h_bar
        else:
            _h_bar = h_bar
        if m == None:
            _m = self.m
        else:
            _m = m
        if V_func == None:
            _V_func = self.V_func
        else:
            _V_func = V_func
        if no == None:
            _n = self.n
        else:
            _n = no

        # Check params
        if _h_bar == None or _m == None or _V_func == None:
            print("Please fill parameters first.")
            return

        # Begin algorithm
        Hamiltonian = np.zeros((_n, _n), dtype=complex)

        schrodinger_scalar = (_h_bar**2) / _m
        schrodinger_scalar_2 = schrodinger_scalar / 2

        if self.debug:
            print("s_s: " + str(schrodinger_scalar))
            print("s_s_2: " + str(schrodinger_scalar_2))

        # Fill first row first
        Hamiltonian[0, 0] = schrodinger_scalar + _V_func(x=0, params=V_params)
        Hamiltonian[0, 1] = -schrodinger_scalar_2

        # Body
        for x in range(1, _n - 1):
            Hamiltonian[x, x - 1] = -schrodinger_scalar_2
            Hamiltonian[x, x] = schrodinger_scalar + _V_func(x=x, params=V_params)
            Hamiltonian[x - 1, x] = -schrodinger_scalar_2

        # Final row
        Hamiltonian[_n - 1, _n - 2] = -schrodinger_scalar_2
        Hamiltonian[_n - 1, _n - 1] = schrodinger_scalar + _V_func(x=x, params=V_params)

        return Hamiltonian

    def populate_hamiltonian(self):
        'Generates a hamiltonian for the class system (time-independent case) and stores it to the class'
        self.Hamiltonian = self.generate_hamiltonian()

        return

    def solve_eigenproblem(self):
        if self.Hamiltonian.all() == None:
            print("Fill hamiltonian first! (use self.populate_hamiltonian)")
            return

        w, v = LA.eig(self.Hamiltonian)

        # Sort
        sort_index = w.argsort()
        self.E = w[sort_index]
        self.Psi = v[:, sort_index]

        return

    def eigen_energies(self):
        if self.E.all() == None:
            print("Call self.solve_eigenproblem() first!")
            return

        return self.E

    def eigen_wavefunctions(self):
        if self.Psi.all() == None:
            print("Call self.solve_eigenproblem() first!")
            return

        return self.Psi

    def check_wavefunctions(self, psis = None, precision = None):
        if psis == None:
            _psis = self.Psi
        else:
            _psis = psis
        if precision == None:
            _precision = 0.01
        else:
            _precision = precision

        if _psis.all() == None:
            print("Please provide wavefunction(s)")
            return

        index = 0
        for psi in _psis:
            sum = 0.0
            for x in psi:
                sum = sum + np.abs(x)**2

            if 1.0 - sum > _precision:
                print("Psi failed at index " + str(index) + " with sum: " + str(sum))

            index = index + 1

    def evolution(self, t_0 = None, t_max = None, delta_t = None, psi_0 = None, hamil_0 = None, V_func = None, evolution_pre_func = None, evolution_post_func = None, step_pre_func = None, step_post_func = None, pre_evol_variables = None, **v_params):
        'Evolve the system according to some passed parameters and using a decorator function. \
        v_params must not include position or time. Passing V_func also gives the option of hard-coding \
        values rather than needing them to be passed through the parameter dictionary. \
        Also contains a lot of decorators so additional values can be calculated or stored without having to \
        rewrite the whole script.'

        # Process arguments
        if psi_0 == None:
            _psi_0 = self.Psi[:, 0]
        else:
            _psi_0 = psi_0
        if hamil_0 == None:
            _hamil_0 = self.Hamiltonian
        else:
            _hamil_0 = hamil_0
        if V_func == None:
            _V_func = self.V_func
        else:
            _V_func = V_func
        if t_0 == None:
            _t_0 = self.t_0
        else:
            _t_0 = t_0
        if t_max == None:
            _t_max = self.t_max
        else:
            _t_max = t_max
        if delta_t == None:
            _delta_t = self.delta_t
        else:
            _delta_t = delta_
        if evolution_pre_func == None:
            _evolution_pre_func = self.evolution_pre_func
        else:
            _evolution_pre_func = evolution_pre_func
        if evolution_post_func == None:
            _evolution_post_func = self.evolution_post_func
        else:
            _evolution_post_func = evolution_post_func
        if step_pre_func == None:
            _step_pre_func = self.step_pre_func
        else:
            _step_pre_func = step_pre_func
        if step_post_func == None:
            _step_post_func = self.step_post_func
        else:
            _step_post_func = step_post_func
        if pre_evol_variables == None:
            _pre_evol_variables = self.pre_evol_variables
        else:
            _pre_evol_variables = pre_evol_variables

        if _hamil_0 == None or _psi_0 == None or _t_0 == None or _t_max == None or _delta_t == None or _V_func == None:
            print("Please populate or supply hamil_0, psi_0, t_0, t_max, delta_t. V_func")
            return

        # Initial condition setup
        no_iterations = int((_t_max - _t_0) / _delta_t)

        self.Psi_vs_t = np.zeros((self.n, no_iterations), dtype=complex)
        self.Psi_vs_t[:, 0] = _psi_0

        curr_hamil = _hamil_0
        next_hamil = None

        t = _t_0
        index = 0

        # Grab an identity matrix
        ID = np.eye(self.n)

        if _evolution_pre_func != None:
            _evolution_pre_func()

        # Evolution (Crank-Nicolson)
        while (t < _t_max):
            # Decorator
            if _step_pre_func != None:
                _step_pre_func()

            # Get the next hamiltonian
            next_hamil = self.generate_hamiltonian(t=(t+_delta_t), V_func=_V_func, params=v_params)

            # Create A
            A = ID + ((1j/2) * _delta_t * next_hamil)

            # Create B
            B = np.matmul((ID - (1j/2) * _delta_t * curr_hamil), self.Psi_vs_t[:, index])

            # Get the next psi
            Psi_vs_t[:, index + 1] = LA.solve(A, B)

            # Call post_function decorator
            if _step_post_func != None:
                _step_post_func()

            # Increment
            curr_hamil = next_hamil

            index = index + 1
            t = t + _delta_t

        if _evolution_post_func != None:
            _evolution_post_func()

def main():
    'For testing'

    def V(x, **params):
        'Test V'
        if x < 50 or x > 150:
            return 0.1
        return 0

    dot = Quantum_Dot(200)

    dot.setter(V_func = V)

    dot.mode_debug()

    dot.populate_hamiltonian()

    dot.solve_eigenproblem()

    dot.check_wavefunctions(precision=0.1)

    from Dot_Plot import simple_plot

    simple_plot(dot.eigen_wavefunctions()[:, 0])


if __name__ == '__main__':
    main()
