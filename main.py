import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
# Assign random values phi_i, h_i

def gen_random_vars(num):
    phi = np.random.uniform(-1.5*np.pi, -0.5*np.pi, size=(num,))
    h = np.random.uniform(-np.pi, np.pi, size=(num,))
    return phi, h

def generate_H_1d(num_qubits, phi, h, g, dnn_coupling=False):
    if dnn_coupling:
        phi2 = np.random.uniform(-1.5 * np.pi, -0.5 * np.pi, size=(num_qubits,))
    identity = [qt.identity(2) for _ in range(num_qubits)]
    ### Rotation by pi*g
    X = [identity.copy() for _ in range(num_qubits)]
    for idx in range(num_qubits):
        X[idx][idx] = (1 / 2) * np.pi * g * qt.sigmax()
        X[idx] = qt.tensor(*(X[idx]))
    H_X = sum(X)
    # print(f"HX: {H_X}")
    ### Ising interaction
    Z = [identity.copy() for _ in range(num_qubits)]
    for idx in range(num_qubits):
        Z[idx][idx] = qt.sigmaz()
        Z[idx] = qt.tensor(*Z[idx])
    H_I = (1 / 4) * sum([phi[idx] * Z[idx] * Z[idx + 1] for idx in range(num_qubits - 1)])
    if dnn_coupling:
        H_I += (1 / 4) * sum([phi2[idx] * Z[idx] * Z[idx + 2] for idx in range(num_qubits - 2)])
    # print(f"HI: {H_I}")
    ### Longitudinal interaction
    H_Z = (1 / 2) * sum([h[idx] * Z[idx] for idx in range(num_qubits)])
    # print(f"H_Z: {H_Z}")
    # Solve for time evolution
    return H_X, H_I, H_Z

def generate_H_2d(axis_length, phi, h, g):
    phi = np.random.uniform(-1.5*np.pi, -0.5*np.pi, size=(2 * axis_length * (axis_length - 1),))
    num_qubits = axis_length**2
    identity = [qt.identity(2) for _ in range(num_qubits)]
    ### Rotation by pi*g
    X = [identity.copy() for _ in range(num_qubits)]
    for idx in range(num_qubits):
        X[idx][idx] = (1 / 2) * np.pi * g * qt.sigmax()
        X[idx] = qt.tensor(*(X[idx]))
    H_X = sum(X)
    # print(f"HX: {H_X}")
    ### Ising interaction
    Z = [identity.copy() for _ in range(num_qubits)]
    for idx in range(num_qubits):
        Z[idx][idx] = qt.sigmaz()
        Z[idx] = qt.tensor(*Z[idx])
    H_I_horz = (1 / 4) * sum([phi[i * (axis_length - 1) + j] * Z[i * axis_length + j] * Z[i * axis_length + (j + 1)]
                              for i in range(axis_length)
                              for j in range(axis_length - 1)])
    H_I_vert = (1 / 4) * sum([phi[axis_length * (axis_length - 1)
                                  + i * axis_length + j] * Z[i * axis_length + j] * Z[(i + 1) * axis_length + j]
                              for i in range(axis_length - 1)
                              for j in range(axis_length)])
    H_I = H_I_horz + H_I_vert
    # print(f"HI: {H_I}")
    ### Longitudinal interaction
    H_Z = (1 / 2) * sum([h[idx] * Z[idx] for idx in range(num_qubits)])
    # print(f"H_Z: {H_Z}")
    # Solve for time evolution
    return H_X, H_I, H_Z

def coupling_H(num_qubits, f, num_crystals=2):
    # Currently just models two Ising spin chains, could be extended to more
    identity = [qt.identity(2) for _ in range(num_crystals * num_qubits)]
    Z = [identity.copy() for _ in range(num_crystals * num_qubits)]
    for idx in range(num_crystals * num_qubits):
        Z[idx][idx] = qt.sigmaz()
        Z[idx] = qt.tensor(*Z[idx])
    H_coup = sum([f * Z[idx] * Z[num_qubits + idx] for idx in range(num_qubits - 1)])
    return H_coup

def simulate_time_crystal_2d(g, axis_length, plot=True):
    num_qubits, num_cycles = axis_length**2, 100
    measure_qubit_idx = axis_length // 2
    observable = [qt.identity(2) for _ in range(num_qubits)]
    observable[measure_qubit_idx * axis_length + measure_qubit_idx] = qt.sigmaz()
    observable = qt.tensor(*observable)
    output = []

    phi, h = gen_random_vars(num_qubits)
    qubits = qt.tensor(*[qt.basis(2, np.random.choice([0, 1])) for _ in range(num_qubits)])
    H_X, H_I, H_Z = generate_H_2d(axis_length, phi, h, g)
    ## Evolve one cycle
    times = [0, 1]
    for cycle_idx in range(num_cycles):
        print(f"Cycle #: {cycle_idx}")
        # result = qt.sesolve(H_X, qubits, times, [observable])
        output_1 = qt.mesolve(H_X, qubits, times, [], []).states[-1]
        output_2 = qt.mesolve(H_I, output_1, times, [], []).states[-1]
        qubits = qt.mesolve(H_Z, output_2, times, [], []).states[-1]
        print(len(qt.mesolve(H_X, qubits, times, [], []).states))
        # Measure observable
        output.append(qt.expect(observable, qubits))

    print(np.mean(np.abs(output)))
    ## Plot observable <Z(t)>
    if plot:
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, num_cycles - 1, num_cycles), output)
        plt.title(r'$<\hat{Z}(t)>$ measured at Central Qubit ($Q_{(2,2)}$) in 2D Case Near $g=g_c$')
        plt.xlabel(r'Number of Cycles $t$')
        plt.ylabel(r'$<\hat{Z}(t)>$')
        plt.show()

    return output

def simulate_time_crystal_1d(g, count=1, plot=True, dnn_coupling=False):
    num_qubits, num_cycles = 11, 100
    # Define observable
    measure_qubit_idx = num_qubits // 2 # 5  # 6th qubit
    observable = [qt.identity(2) for _ in range(count * num_qubits)]
    observable[measure_qubit_idx] = qt.sigmaz()
    observable = qt.tensor(*observable)
    output = []
    # Setup
    ## Sample random vars
    phi, h = gen_random_vars(count * num_qubits)

    ## Define qubits and Hamiltonians

    qubits = qt.tensor(*[qt.basis(2, np.random.choice([0, 1])) for _ in range(count * num_qubits)])
    H_X, H_I, H_Z = generate_H_1d(num_qubits, phi, h, g, dnn_coupling)
    if count > 1:
        H_X, H_I, H_Z = [], [], []
        for crystal_idx in range(count):
            H_X_i, H_I_i, H_Z_i = generate_H_1d(num_qubits, phi, h, g, dnn_coupling)
            H_X.append(H_X_i.copy())
            H_I.append(H_I_i.copy())
            H_Z.append(H_Z_i.copy())
        H_X = qt.tensor(H_X[0], qt.qeye(2**num_qubits)) + qt.tensor(qt.qeye(2**num_qubits), H_X[0])
        H_I = qt.tensor(H_I[0], qt.qeye(2**num_qubits)) + qt.tensor(qt.qeye(2**num_qubits), H_I[0])
        H_Z = qt.tensor(H_Z[0], qt.qeye(2**num_qubits)) + qt.tensor(qt.qeye(2**num_qubits), H_Z[0])
        H_coup = coupling_H(num_qubits, f=0.0)
    ## Evolve one cycle
    times = [0, 1]
    for cycle_idx in range(num_cycles):
        print(f"Cycle #: {cycle_idx}")
        # result = qt.sesolve(H_X, qubits, times, [observable])
        output_1 = qt.mesolve(H_X, qubits, times, [], []).states[-1]
        output_2 = qt.mesolve(H_I, output_1, times, [], []).states[-1]
        # if count > 1:
        #     output_2 = qt.mesolve(H_coup, output_1, times, [], []).states[-1]
        qubits = qt.mesolve(H_Z, output_2, times, [], []).states[-1]
        # print(len(qt.mesolve(H_X, qubits, times, [], []).states))
        # Measure observable
        output.append(qt.expect(observable, qubits))

    print(np.mean(np.abs(output)))
    ## Plot observable <Z(t)>
    if plot:
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, num_cycles - 1, num_cycles), output)
        plt.title(r'$<\hat{Z}(t)>$ measured at Central Qubit ($Q_6$) with $g=0.97$ (MBL-DTC Regime)')
        plt.xlabel(r'Number of Cycles $t$')
        plt.ylabel(r'$<\hat{Z}(t)>$')
        plt.show()

    return output

def main():
    # Binary search to find g threshold
    low, high = 0, 1
    g = (low + high) / 2
    num_iterations = 10
    g_thermal, g_MBL_DTC = 0.6, 0.97  # cutoff at 0.84
    # # g = g_MBL_DTC
    g = g_MBL_DTC
    # output = simulate_time_crystal_1d(g, count=1, dnn_coupling=False, plot=True)
    # output = simulate_time_crystal_2d(g, axis_length=3, plot=True)
    for idx in range(num_iterations):
        # Experiments:
        output = simulate_time_crystal_2d(g, axis_length=3, plot=True)
        # output = simulate_time_crystal_1d(g, plot=True)
        # output = simulate_time_crystal_1d(g, count=1, dnn_coupling=False, plot=False)
        # slight allowance if simulation ends on pi/2 rotation from initial state
        if np.mean(np.abs(output)) > 0.5:
            ## This was a DTC, let's try lower g
            high = g
            g = (low + high) / 2
            print(f"g: {g}")
        else:
            ## This thermalized, let's try higher g
            low = g
            g = (low + high) / 2
            print(f"g: {g}")
    print(f"g boundary: {g}")

if __name__ == '__main__':
    main()