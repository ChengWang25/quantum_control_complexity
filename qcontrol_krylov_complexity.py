import numpy as np
from scipy.linalg import expm
import sys



# --- 1. Operator Definitions ---

# Define the 2x2 Pauli matrices as dense matrices
PAULI = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex)
}

def pauli_string_to_matrix(p_string: str) -> np.ndarray:
    """
    Converts a Pauli string like 'IXYZ' into its 2^N x 2^N matrix representation.

    Args:
        p_string: A string of 'I', 'X', 'Y', 'Z' characters.

    Returns:
        A numpy.ndarray representing the tensor product.
    """
    N = len(p_string)
    if N == 0:
        return np.array([[1.0]], dtype=complex)

    # Start with the first matrix
    M = PAULI[p_string[0]]

    # Kronecker product with the rest
    for char in p_string[1:]:
        M = np.kron(M, PAULI[char])
    return M
    

def build_tfim_hamiltonian(N: int, J: float, g: float) -> np.ndarray:
    """
    Builds the Hamiltonian for the Transverse Field Ising Model (TFIM).
    H = -J * sum(Z_i Z_{i+1}) - g * sum(X_i)

    Args:
        N: Number of spins.
        J: ZZ coupling strength.
        g: Transverse field strength.

    Returns:
        The Hamiltonian as a numpy.ndarray.
    """
    H = np.zeros((2**N, 2**N), dtype=complex)

    # ZZ interaction terms
    for i in range(N):
        p_list = ['I'] * N
        p_list[i] = 'Z'
        p_list[(i + 1) % N] = 'Z' # Periodic boundary conditions
        H -= J * pauli_string_to_matrix("".join(p_list))

    # X field terms
    for i in range(N):
        p_list = ['I'] * N
        p_list[i] = 'X'
        H -= g * pauli_string_to_matrix("".join(p_list))

    return H

def build_ZZ_interaction(N: int, J: float) -> np.ndarray:
    """
    Builds the ZZ interaction term for the TFIM.
    """
    H = np.zeros((2**N, 2**N), dtype=complex)
    
    for i in range(N - 1):
        p_list = ['I'] * N
        p_list[i] = 'Z'
        p_list[i + 1] = 'Z'
        H -= J * pauli_string_to_matrix("".join(p_list))

    return H

def build_Z_field(N: int, g_Z: float) -> np.ndarray:
    """
    Builds the Z field term for the TFIM.
    """
    H = np.zeros((2**N, 2**N), dtype=complex)

    for i in range(N):
        p_list = ['I'] * N
        p_list[i] = 'Z'
        H -= g_Z * pauli_string_to_matrix("".join(p_list))

    return H


def build_X_field(N: int, g_X: float) -> np.ndarray:
    """
    Builds the X field term for the TFIM.
    """
    H = np.zeros((2**N, 2**N), dtype=complex)

    for i in range(N):
        p_list = ['I'] * N
        p_list[i] = 'X'
        H -= g_X * pauli_string_to_matrix("".join(p_list))
    
    return H


def build_X_odd_field(N: int, g_X_odd: float) -> np.ndarray:
    """
    Builds the X field term on odd sites for the TFIM.
    """
    H = np.zeros((2**N, 2**N), dtype=complex)
    
    for i in range(1, N, 2):
        p_list = ['I'] * N
        p_list[i] = 'X'
        H -= g_X_odd * pauli_string_to_matrix("".join(p_list))

    return H

def build_X_even_field(N: int, g_X_even: float) -> np.ndarray:
    """
    Builds the X field term on even sites for the TFIM.
    """
    H = np.zeros((2**N, 2**N), dtype=complex)
    
    for i in range(0, N, 2):
        p_list = ['I'] * N
        p_list[i] = 'X'
        H -= g_X_even * pauli_string_to_matrix("".join(p_list))

    return H



# --- 2. Operator Space Definitions ---

def inner_product(A: np.ndarray, B: np.ndarray, N: int) -> float:
    """
    Computes the normalized Hilbert-Schmidt inner product:
    <A|B> = (1/2^N) * Tr(A^\dagger B)

    Args:
        A, B: The two operators (as dense matrices).
        N: Number of spins.

    Returns:
        The inner product (should be real).
    """
    trace = np.vdot(A, B)
    return trace / (2**N)


def apply_liouvillian(H: np.ndarray, O: np.ndarray) -> np.ndarray:
    """
    Applies the Liouvillian superoperator L(O) = [H, O].

    Args:
        H: The system Hamiltonian.
        O: The operator to act on.

    Returns:
        The resulting operator [H, O] as a dense matrix.
    """
    return H @ O - O @ H

def schmidt_orthogonalize(O_old: np.ndarray, O_new: np.ndarray, N: int) -> np.ndarray:
    """
    Orthogonalizes the new operator O_new over the old operator O_old.
    """
    return O_new - inner_product(O_old, O_new, N) / inner_product(O_old, O_old, N) * O_old

def normalize(O: np.ndarray, N: int) -> np.ndarray:
    """
    Normalizes the operator O.
    """
    return O / np.sqrt(inner_product(O, O, N))



# --- 3. Lanczos Algorithm ---

def lanczos_algorithm(H_set: list[np.ndarray], O_initial_set: list[np.ndarray]):
    """
    Performs the Lanczos algorithm to find the local Hausdorff dimension and the steps of the algorithm.

    Args:
        H_set: The controlled Hamiltonians.
        O_initial: The starting operator O_0.
        K_max: The maximum number of Lanczos steps .

    Returns:
        The steps of the Lanczos algorithm and local Hausdorff dimension.
    """
    N = int(np.log2(H_set[0].shape[0]))
    K_max = 4 ** N - 1

    steps = 0 # steps of the algorithm
    dimension = 0 # local Hausdorff dimension
    operator_count = 0 # number of operators in the basis
    O_basis = [[]] # memory of current operator basis

    # --- Initialization (n=0) ---
    # Orthonormalize the initial operator
    for i in range(len(O_initial_set)):
        if len(O_basis[0]) == 0 :
            O_initial = O_initial_set[i]
            norm_O_sq = inner_product(O_initial, O_initial, N)
            O_n = O_initial / np.sqrt(norm_O_sq)
            O_basis[0].append(O_n) 
        else :
            O_n = O_initial_set[i]
            for j in range(len(O_basis[0])):
                O_n = schmidt_orthogonalize(O_basis[0][j], O_n, N)
            norm_O_sq = inner_product(O_n, O_n, N)
            if norm_O_sq > 1e-12:
                O_n = O_n / np.sqrt(norm_O_sq)
                O_basis[0].append(O_n) 
    
    steps += 1
    operator_count += len(O_basis[0])
    dimension += len(O_basis[0]) * steps

    # --- Main Loop ---
    while operator_count < K_max and len(O_basis[-1]) > 0:
        # Apply Liouvillian
        O_basis.append([])
        for k in range(len(O_basis[-2])):
            for H in H_set:
                L_On = apply_liouvillian(H, O_basis[-2][k])
                if inner_product(L_On, L_On, N) > 1e-12:
                    # Orthogonalize over past basis
                    for i in range(len(O_basis)):
                        if len(O_basis[i]) > 0:
                            for j in range(len(O_basis[i])):
                                L_On = schmidt_orthogonalize(O_basis[i][j], L_On, N)
                    if inner_product(L_On, L_On, N) > 1e-12:
                        L_On = normalize(L_On, N)
                        O_basis[-1].append(L_On)
                            

        # throw away the very first basis
        if len(O_basis) > 2:
            O_basis.pop(0)

        # count the number of new basis
        steps += 1
        operator_count += len(O_basis[-1])
        dimension += len(O_basis[-1]) * steps

    return steps, operator_count, dimension



# --- 4. Main Execution ---

def main():
    """
    Main function to run the calculation.
    """
    # --- Parameters ---
    N_list = list(range(2, 6, 1))           # Number of spins 
    J = 1.0         # ZZ coupling
    g_Z = 1.0         # Z field
    g_X_odd = 1.0   # X field on odd sites
    g_X_even = 1.0  # X field on even sites
    steps_list = []
    operator_count_list = []
    dimension_list = []

    for N in N_list:

        print("Current number of spins:")
        print(f"N = {N}")

        # --- Build Operators ---
        print("Building controlled Hamiltonian set H...")
        H_set = [build_ZZ_interaction(N, J), build_Z_field(N, g_Z), build_X_odd_field(N, g_X_odd), build_X_even_field(N, g_X_even), pauli_string_to_matrix("".join(['I'] * (N-1) + ['X']))]

        print("Building initial operator O(0)...")
        O_initial = H_set.copy()

        # --- Run Lanczos ---
        print("Running Lanczos algorithm...")
        steps, operator_count, dimension = lanczos_algorithm(H_set, O_initial)
        steps_list.append(steps)
        operator_count_list.append(operator_count)
        dimension_list.append(dimension)

        print(f"steps: {steps}, operator_count: {operator_count}, dimension: {dimension}")

    print(f"steps_list: {steps_list}")
    print(f"operator_count_list: {operator_count_list}")
    print(f"dimension_list: {dimension_list}")

    

if __name__ == "__main__":
    main()