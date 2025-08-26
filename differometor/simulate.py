import jax
from time import time
from tqdm import tqdm
import jax.numpy as jnp
from jax import lax, vmap, jit
from differometor.setups import Setup
from differometor.build import build, pairs_to_arrays, prepare_arrays, LINKING_FUNCTIONS
from differometor.components import FUNCTIONS, UNIT_VACUUM, H_PLANCK, F0


jax.config.update("jax_enable_x64", True)


def solve(matrix, right_hand_side) -> jnp.ndarray:
    """
    system_matrix has dimension (range, system_size, system_size + 1) where range
    is the number of values that the changing component takes, system_size is the
    size of the system matrix and the plus one is the input vector.
    """
    # right_hand_side needs to be jnp.complex128 and matrix needs to be jnp.complex64 to 
    # avoid numerical errors. See linalg_solve_debug.ipynb for more information.
    right_hand_side = right_hand_side.astype(jnp.complex128)
    matrix = matrix.astype(jnp.complex128)

    # jnp.angle here throws nan gradients for an unknown reason. That's why
    # we return the solution directly without any additional processing like
    # transforming it into phases, amplitudes, and powers.
    return jnp.linalg.solve(matrix, right_hand_side)


def update(
        parameters, 
        matrix, 
        function_input_indices,
        output_indices,
        matrix_indices,
        function_indices
    ):
    # select the parameters that are used as inputs to the functions (e.g. loss, reflectivity, tuning for mirror)
    function_inputs = parameters[function_input_indices]

    # Function list as implicit global parameter: https://stackoverflow.com/questions/73621269/jax-jitting-functions-parameters-vs-global-variables#:~:text=During%20JIT%20tracing%2C%20JAX%20treats,the%20jaxpr%20representing%20the%20function.
    output_matrix = vmap(lambda i, x: lax.switch(i, FUNCTIONS, x))(function_indices, function_inputs)
    outputs = output_matrix[output_indices[0], output_indices[1]]
    matrix = matrix.at[matrix_indices[0], matrix_indices[1]].set(outputs)
    return matrix


def expand_parameters(
        array, 
        indices, 
        values
    ):
    # array has to have shape (1, N), indices has to have shape (V, )
    # values for each index have to be stacked row wise and must have dimension (V, R)
    # e.g. array = jnp.array([[1, 2, 3, 4, 5]]), indices = jnp.array([0, 1]), 
    # values = jnp.array([[12, 14], [16, 18]]) results in 
    # jnp.array([[12, 16, 3, 4, 5], [14, 18, 3, 4, 5]
    tiled_array = jnp.tile(array, (values.shape[1], 1))
    row_indices = jnp.repeat(jnp.arange(values.shape[1]), len(indices))
    column_indices = jnp.tile(indices, values.shape[1])
    flat_values = values.T.flatten()
    return tiled_array.at[row_indices, column_indices].set(flat_values)


# This function has to be jittable
def simulate_in_parallel(
            # prepared arrays
            optimized_parameters,
            optimized_parameter_indices,
            optimized_value_indices,
            carrier_changing_parameter_indices,
            carrier_changing_values,
            signal_changing_parameter_indices,
            signal_changing_values,
            # matrices
            parameters,
            carrier_matrix,
            signal_matrix,
            noise_matrix,
            noise_selection_vectors,
            qhd_parameter_indices,
            qhd_placing_indices,
            linked_indices,
            linking_function_indices,
            indices_to_link,
            # carrier arrays
            carrier_function_input_indices,
            carrier_function_indices,
            carrier_output_indices,
            carrier_matrix_indices,
            # signal arrays
            signal_function_input_indices,
            signal_function_indices,
            signal_output_indices,
            signal_matrix_indices,
            signal_carrier_indices,
            signal_carrier_placing_indices,
            # noise arrays
            noise_function_input_indices,
            noise_function_indices,
            noise_output_indices,
            noise_system_matrix_indices
        ):
        """
        
        Returns
        -------
        carrier: jnp.ndarray
            Shape (carrier_value_length, port_number). The solution of the carrier system.
        signal: jnp.ndarray
            Shape (signal_value_length, port_number). The solution of the signal system.
        noise: jnp.ndarray
            Shape (detector_number, signal_value_length). The solution of the noise system.
        """
        # insert the parameter values that actually get optimized
        # parameters shape: (1, P), optimized_parameter_indices shape: (OP), parameters_to_optimize shape: (OP)       
        parameters = parameters.at[[0], optimized_parameter_indices].set(optimized_parameters[optimized_value_indices])
        # parameters shape does not change and is still (1, P)

        inputs_to_link = parameters[0, linked_indices]
        outputs_to_link = vmap(lambda i, x: lax.switch(i, LINKING_FUNCTIONS, x))(linking_function_indices, inputs_to_link)
        parameters = parameters.at[[0], indices_to_link].set(outputs_to_link)

        # Must be done before parameter expansion
        qhd_phase_values = jnp.exp(1j * jnp.radians(parameters[0, qhd_parameter_indices]))

        # --- CARRIER SOLVING ---
        # changing_parameter_indices shape: (CCP), values shape: (CCP, CV)
        parameters = expand_parameters(parameters, carrier_changing_parameter_indices, carrier_changing_values)
        # parameters gets expanded to shape (CV, P)

        # carrier matrix shape: (CN, CN + 1)
        carrier_matrix = vmap(update, in_axes=(0, None, None, None, None, None))(
                                parameters,
                                carrier_matrix,
                                carrier_function_input_indices,
                                carrier_output_indices,
                                carrier_matrix_indices,
                                carrier_function_indices)
        # carrier matrix gets expanded to shape (CV, CN, CN + 1)

        # solve the carrier system, carrier_matrix_shape: (CV, CN, CN + 1), so we vmap over the first dimension in both
        carrier = vmap(solve)(carrier_matrix[:, :, :-1], carrier_matrix[:, :, -1])
        # carrier_solution shape: (CV, CN)

        # --- SIGNAL SOLVING ---
        # (not in lax cond because we would then have to compile one conditional path with an
        # empty array which would throw an error)
        # In this design the signal path gets executed independent from if there is a signal or 
        # not. This is done to avoid conditionals, but of course it introduces some computational
        # overhead.
        # parameters shape: (CV, P), signal_changing_parameter_indices shape: (SCP), signal_values shape: (SCP, SV)
        parameters = expand_parameters(parameters, signal_changing_parameter_indices, signal_changing_values)
        # parameters gets expanded to shape (SV, P), CV = 1 when SV > 1

        # signal matrix shape: (SN, SN + 1)
        signal_matrix = vmap(update, in_axes=(0, None, None, None, None, None))(
                             parameters, 
                             signal_matrix, 
                             signal_function_input_indices, 
                             signal_output_indices, 
                             signal_matrix_indices, 
                             signal_function_indices)
        # signal matrix gets expanded to shape (SV, SN, SN + 1) 

        # multiply the signal columns that contain the signal connector entries (not the right hand side)
        # with carrier solution (see https://arxiv.org/abs/1306.6752). Signals just scale the carrier solution
        # at the respective spaces.
        # Notice the minus sign for the signal entries. This is to conform with Finesse 3
        signal_entries = - signal_matrix[:, signal_carrier_placing_indices[0], signal_carrier_placing_indices[1]] * carrier[:, signal_carrier_indices.flatten()]
        signal_matrix = signal_matrix.at[:, signal_carrier_placing_indices[0], signal_carrier_placing_indices[1]].set(signal_entries)

        # apply conjugation
        carrier_size = carrier.shape[1]
        # first the component part of the lower sideband (e.g. mirrors, spaces)
        signal_matrix = signal_matrix.at[:, carrier_size:carrier_size * 2, carrier_size:carrier_size * 2].set(jnp.conjugate(signal_matrix[:, carrier_size:carrier_size * 2, carrier_size:carrier_size * 2]))
        # the column signals of the upper sideband (e.g. force connectors)
        signal_matrix = signal_matrix.at[:, carrier_size*2:, :carrier_size].set(jnp.conjugate(signal_matrix[:, carrier_size*2:, :carrier_size]))
        # the row signals of the lower sideband (e.g. signal connectors)
        signal_matrix = signal_matrix.at[:, carrier_size:carrier_size * 2, carrier_size*2:].set(jnp.conjugate(signal_matrix[:, carrier_size:carrier_size * 2, carrier_size*2:]))

        # solve the signal system, signal_matrix_shape: (SV, SN, SN+1), so we vmap over the first dimension in both                                                                                              
        signal = vmap(solve)(signal_matrix[:, :, :-1], signal_matrix[:, :, -1])
        # signal_solution shape: (SV, SN)

        # --- QNOISE SOLVING ---
        # refer to https://doi.org/10.5281/zenodo.821380 appendix D about quantum noise and to
        # finesse > simulations > sparse > KLU.pyx > solve_noises()

        # using scatter_mul ensures that code also works with empty arrays without using conditionals
        # normal .at .set does not work with empty arrays
        noise_selection_vectors = lax.scatter_mul(
             noise_selection_vectors,
             qhd_placing_indices.reshape(-1, 3),
             qhd_phase_values,
             lax.ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0, 1, 2),
                    scatter_dims_to_operand_dims=(0, 1, 2)
                ),
            unique_indices=True,
        )

        # noise_selection_vectors shape: (detector_number, 1, setup_size)
        noise_selection_vectors = jnp.tile(noise_selection_vectors, (1, carrier.shape[0], 1))

        # select necessary carrier solution entries
        # upper sideband
        upper_carrier_selections = noise_selection_vectors[:, :, :carrier.shape[1]] * carrier
        upper_carrier_selections = noise_selection_vectors.at[:, :, :carrier.shape[1]].set(upper_carrier_selections)
        # lower sideband
        lower_carrier_selections = jnp.conjugate(upper_carrier_selections[:, :, carrier.shape[1]:carrier.shape[1]*2] * carrier)
        carrier_selections = upper_carrier_selections.at[:, :, carrier.shape[1]:carrier.shape[1]*2].set(lower_carrier_selections)

        # propagate the weights backwards through the noise matrix by solving with transposed and conjugated signal matrix 
        # only take the first n-1 columns of the signal matrix because the last column is the right hand side
        # only transpose last two dimensions because 
        transposed_conjugated_signal_matrix = jnp.transpose(jnp.conjugate(signal_matrix[:, :, :-1]), (0, 2, 1))

        # transposed_conjugated_signal_matrix shape: (value_length, setup_size, setup_size)
        # carrier_selections shape: (detector_number, 1, setup_size)
        carrier_selections = jnp.broadcast_to(carrier_selections, (carrier_selections.shape[0], transposed_conjugated_signal_matrix.shape[0], carrier_selections.shape[2]))

        # noise_weights shape: (detector_number, value_length, setup_size)
        noise_weights = vmap(vmap(solve), in_axes=(None, 0))(transposed_conjugated_signal_matrix, carrier_selections)

        # TODO: ideally we would only do this update if it is necessary because the parameters indicated
        # by noise_function_input_indices have changed. Otherwise we could just use jnp.tile
        noise_matrix = vmap(update, in_axes=(0, None, None, None, None, None))(
                            parameters, 
                            noise_matrix, 
                            noise_function_input_indices, 
                            noise_output_indices, 
                            noise_system_matrix_indices, 
                            noise_function_indices)

        def matmul_single_batch(matrix, vector):
            return jnp.einsum('bij,bj->bi', matrix, vector)

        # noise_matrix shape: (V, N, N), noise_weights shape: (detector_number, V, N)
        # vmap over the detectors
        noise_sources = vmap(matmul_single_batch, in_axes=(None, 0))(noise_matrix, noise_weights)

        # propagate the noise sources through the signal matrix (vmap over the detectors)
        covariances = vmap(vmap(solve), in_axes=(None, 0))(signal_matrix[:, :, :-1], noise_sources)

        # see finesse > detectors > compute > quantum.pyx > c_qnd0_output()
        temporary_sum = jnp.sum(jnp.real(covariances * jnp.conjugate(carrier_selections)), axis=2)
        # noise_solution shape: (detector_number, V)
        # square root because we return ASD instead of PSD, 0.25 for demodulation (see quantum_noise_detector.py > 
        # QuantumNoiseDetector documentation where it states that there is one demodulation at the signal frequency), 
        # 2, f, unit_vacuum and h_planck from the Schottky formula
        noise = jnp.sqrt(2 * temporary_sum * UNIT_VACUUM * H_PLANCK * F0 * 0.25)

        return carrier.T, signal.T, noise.squeeze()


def run_build_step(
        setup: Setup,
        changing_pairs: list = None,
        changing_values: jnp.ndarray = None,
        optimization_pairs: list = None,
        timeit: bool = False
    ) -> tuple[tuple, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Builds the setup.

    Parameters
    ----------
    setup: Setup object
        The setup to be simulated.
    changing_pairs: list of tuples
        Each tuple contains the name of the component and the name of the parameter to be changed
    changing_values: jnp.ndarray
        Values for the parameters to be changed
    optimization_pairs: list of tuples
        Each tuple contains the name of the component and the name of the parameter to be optimized
    timeit: bool
        If True, the function will return the time it took to build the setup
        
    Returns
    -------
    simulation_arrays: tuple of jnp.ndarray
        The arrays needed for the simulation
    detector_indices: jnp.ndarray
        Port indices of the detectors in the order defined in the setup
    mirror_indices: jnp.ndarray
        Port indices of the mirrors in the order defined in the setup
    beamsplitter_indices: jnp.ndarray
        Port indices of the beamsplitters in the order defined in the setup
    isolator_indices: jnp.ndarray
        Port indices of the isolators in the order defined in the setup
    """
    start = time()
    instructions, matrices, metadata = build(setup)
    arrays_to_prepare, carrier_arrays, signal_arrays, noise_arrays = pairs_to_arrays(*instructions, optimization_pairs, changing_pairs)
    prepared_arrays = prepare_arrays(*arrays_to_prepare, changing_values, parameters=matrices[0])

    simulation_arrays = (*prepared_arrays, *matrices, *carrier_arrays, *signal_arrays, *noise_arrays)
    
    if timeit:
        for array in simulation_arrays:
            jax.block_until_ready(array)
        return time() - start, (simulation_arrays, metadata)
    return simulation_arrays, *metadata


def run_simulation_step(
        simulation_arrays,
        jit_simulation: bool = False,
        timeit: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Runs the simulation. Usually comes after the run_build_step function.

    Parameters
    ---------- 
    simulation_arrays: tuple of jnp.ndarray
        The arrays needed for the simulation. This is the output of the run_build_step function
    jit_simulation: bool
        If True, the function will be jitted and include a warmup run
    timeit: bool
        If True, the function will return the time it took to run the simulation
    
    Returns
    -------
    carrier: jnp.ndarray
        Shape (carrier_value_length, port_number). The solution of the carrier system.
    signal: jnp.ndarray
        Shape (signal_value_length, port_number). The solution of the signal system.
    noise: jnp.ndarray
        Shape (detector_number, signal_value_length). The solution of the noise system.
    """

    simulation_function = simulate_in_parallel

    if jit_simulation:
        # warmup
        simulation_function = jit(simulate_in_parallel)
        simulation_function(*simulation_arrays)

    start = time()
    results = simulation_function(*simulation_arrays) 

    if timeit:
        for result in results:
            jax.block_until_ready(result)
        return time() - start, results
    return results


def run(
        setup: Setup,
        changing_pairs: list = None,
        changing_values: jnp.ndarray = None,
        optimization_pairs: list = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Builds the setup and runs the simulation.

    Parameters
    ----------
    setup: Setup 
        The setup to be simulated.
    changing_pairs: list of tuples
        Each tuple contains the name of the component and the name of the parameter to be changed
    changing_values: jnp.ndarray
        values for the parameters to be changed
    optimization_pairs: list of tuples
        Each tuple contains the name of the component and the name of the parameter to be optimized
    
    Returns
    -------
    carrier: jnp.ndarray
        Shape (carrier_value_length, port_number). The solution of the carrier system.
    signal: jnp.ndarray
        Shape (signal_value_length, port_number). The solution of the signal system.
    noise: jnp.ndarray
        Shape (detector_number, signal_value_length). The solution of the noise system.
    detector_indices: jnp.ndarray
        Port indices of the detectors in the order defined in the setup.
    mirror_indices: jnp.ndarray
        Port indices of the mirrors in the order defined in the setup.
    beamsplitter_indices: jnp.ndarray
        Port indices of the beamsplitters in the order defined in the setup.
    isolator_indices: jnp.ndarray
        Port indices of the isolators in the order defined in the setup.
    """
    simulation_arrays, *metadata = run_build_step(setup, changing_pairs, changing_values, optimization_pairs)
    carrier, signal, noise = run_simulation_step(simulation_arrays) 
    
    return carrier, signal, noise, *metadata


def run_setups(
        setups, 
        frequencies
    ):
    results = []
    for setup in setups:
        result = run(setup, [("f", "frequency")], frequencies)
        results.append(result)
    return results


def run_with_parameter_sets(
        setup, 
        parameter_sets_to_run: jnp.ndarray, 
        parameter_set_pairs: list, 
        changing_pairs: list = None, 
        changing_parameter_values = None,
    ):
    simulation_arrays, *metadata = run_build_step(setup, changing_pairs, changing_parameter_values, parameter_set_pairs)

    def run_simulation(parameter_set_to_run):
        carrier, signal, noise = simulate_in_parallel(parameter_set_to_run, *simulation_arrays[1:])
        return carrier, signal, noise
    
    jitted_run_simulation = jit(run_simulation)

    carriers = []
    signals = []
    noises = []
    for ix in tqdm(range(parameter_sets_to_run.shape[0])):
        carrier_solution, signal_solution, noise_solution = jitted_run_simulation(parameter_sets_to_run[ix])

        carriers.append(jnp.expand_dims(carrier_solution, axis=-1))
        signals.append(jnp.expand_dims(signal_solution, axis=-1))
        noises.append(jnp.expand_dims(noise_solution, axis=-1))

    return jnp.concatenate(carriers, axis=-1), jnp.concatenate(signals, axis=-1), jnp.concatenate(noises, axis=-1), *metadata


def run_setups_with_parameter_sets(
        setups, 
        frequencies, 
        parameter_sets_to_run, 
        bounds,
        parameter_set_pairs,
        bounding_function
    ):
    parameter_sets_to_run = bounding_function(parameter_sets_to_run, bounds)
    results = []
    for setup in setups:
        result = run_with_parameter_sets(setup, 
                                          parameter_sets_to_run, 
                                          parameter_set_pairs, 
                                          [("f", "frequency")], 
                                          frequencies)
        results.append(result)
    return results
