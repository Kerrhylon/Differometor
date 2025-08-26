import os
import numpy as np
import jax.numpy as jnp
from differometor.plot import plot_powers, plot_comparison
from differometor.simulate import run_setups, run_setups_with_parameter_sets
from differometor.components import power_detector, demodulate_signal_power


def default_bounding(
        parameters, 
        bounds
    ):
    return parameters * (bounds[1] - bounds[0]) + bounds[0]


def clip_bounding(
        parameters, 
        bounds
    ):
    epsilon = 1e-10
    clipped_parameters = jnp.clip(parameters, 0.0 + epsilon, 1.0 - epsilon)
    return clipped_parameters * (bounds[1] - bounds[0]) + bounds[0]


def tanh_bounding(
        parameters, 
        bounds
    ):
    tanh_parameters = 0.5 * (jnp.tanh(parameters) + 1.0)
    return tanh_parameters * (bounds[1] - bounds[0]) + bounds[0]


def sigmoid_bounding(
        parameters, 
        bounds
    ):
    sigmoid_parameters = 1 / (1 + jnp.exp(-parameters))
    return sigmoid_parameters * (bounds[1] - bounds[0]) + bounds[0]


def set_value(
        node, 
        property_name, 
        value, 
        setup
    ):
    if not '_' in node:
        try:
            setup.nodes[node]["properties"][property_name] = value
        except KeyError:
            setup.nodes[node]["properties"] = {property_name: value}
    else:
        source, target = node.split('_')
        try:
            setup.edges[source + '_' + target]["properties"][property_name] = value
        except KeyError:
            setup.edges[source + '_' + target]["properties"] = {property_name: value}


def calculate_powers(
        carrier_solution,
        detector_indices,
        mirror_indices,
        beamsplitter_indices,
        isolator_indices
    ):
    """
    Calculates the powers at all detectors, mirrors, beamsplitters and isolators defined by the provided indices.

    Parameters
    ----------
    carrier_solution: jnp.ndarray 
        The solution of the carrier system.
    detector_indices: jnp.ndarray
        The indices of all detectors within the setup.
    mirror_indices: jnp.ndarray
        The indices of all mirrors within the setup.
    beamsplitter_indices: jnp.ndarray
        The indices of all beamsplitters within the setup.
    isolator_indices: jnp.ndarray
        The indices of all isolators within the setup.

    Returns
    -------
    hard_side_powers: jnp.ndarray
        The maximal powers at the component sides of maximal power
    soft_side_powers: jnp.ndarray
        The maximal powers at the component sides of minimal power
    detector_powers: jnp.ndarray
        The powers at the detector ports
    """
    # beamsplitter left in out, top in out, right in out, bottom in out
    # mirror left in out, right in out
    # 1 beamsplitter, 6 mirrors => 1 * 8 + 6 * 4 = 32 ports
    carrier_powers = power_detector(carrier_solution)
    beamsplitter_powers = carrier_powers[beamsplitter_indices]
    mirror_powers = carrier_powers[mirror_indices]
    isolator_powers = carrier_powers[isolator_indices]

    beamsplitters = beamsplitter_powers.reshape(-1, 8, *beamsplitter_powers.shape[1:])  # Reshape beamsplitters (x pack of 8 ports)
    # For the beam splitter: Identify the port with maximum power on each side
    # Sides: left+top (ports 0-3), right+bottom (ports 4-7)
    beamsplitter_left_top = beamsplitters[:, :4]
    beamsplitter_right_bottom = beamsplitters[:, 4:]

    # Maximum power and port index for each side
    max_power_left_top = jnp.max(beamsplitter_left_top, axis=1)
    max_power_right_bottom = jnp.max(beamsplitter_right_bottom, axis=1)

    # Identify the overall maximum power and its opposite side power
    if_beamsplitter = max_power_left_top > max_power_right_bottom
    max_beamsplitter_power = jnp.where(if_beamsplitter, max_power_left_top, max_power_right_bottom)
    opposite_beamsplitter_power = jnp.where(if_beamsplitter, max_power_right_bottom, max_power_left_top)

    mirrors = mirror_powers.reshape(-1, 4, *mirror_powers.shape[1:])  # Reshape mirrors (y packs of 4 ports)
    # For mirrors: Compare individual ports
    # Sides: left (ports 0-1 for each pack), right (ports 2-3 for each pack)
    mirrors_left = mirrors[:, :2]
    mirrors_right = mirrors[:, 2:]

    # Maximum power on each side for all packs
    max_power_left = jnp.max(mirrors_left, axis=1)
    max_power_right = jnp.max(mirrors_right, axis=1)

    # Identify the maximum power across sides and the opposite side's maximum
    if_mirrors = max_power_left > max_power_right
    max_mirrors_power = jnp.where(if_mirrors, max_power_left, max_power_right)
    opposite_mirrors_power = jnp.where(if_mirrors, max_power_right, max_power_left)

    # Combine results into the required arrays
    hard_side_powers = jnp.concatenate([max_beamsplitter_power, max_mirrors_power], axis=0)    
    soft_side_powers = jnp.concatenate([opposite_beamsplitter_power, opposite_mirrors_power, isolator_powers], axis=0)
    detector_powers = carrier_powers[detector_indices]

    return hard_side_powers, soft_side_powers, detector_powers


def update_setup(
        parameters, 
        optimization_pairs, 
        bounds, 
        setup,
        bounding_function=sigmoid_bounding
    ):
    for ix, optimization_pair in enumerate(optimization_pairs):
        value = float(bounding_function(parameters[ix], bounds[:, ix]))
        if isinstance(optimization_pair[0], list):
            for component_name, property_name in optimization_pair:
                set_value(component_name, property_name, value, setup)
        else:
            component_name, property_name = optimization_pair
            set_value(component_name, property_name, value, setup)


def calculate_sensitivities(
        results,
        sensitivity_function,
        homodyne=True
    ):
    """
    Calculates the sensitivities of the setup.

    Parameters
    ----------
    results: list
        A list of the following structure: [(carrier, signal, noise, detector_indices, mirror_indices, beamsplitter_indices, isolator_indices)]
    sensitivity_function: callable
        A function that calculates the sensitivity given the noise and power levels.
    homodyne: bool
        Whether the setup uses a balanced homodyne detection scheme or not.

    Returns
    -------
    sensitivities: jnp.ndarray
        The calculated sensitivities for the setup.
    """
    noises = []
    powers = []
    for result in results:
        signal_powers = demodulate_signal_power(result[0], result[1])
        signal_powers = signal_powers[result[3]]
        if homodyne:
            # assuming only two detectors
            signal_powers = jnp.abs(signal_powers[0] - signal_powers[1]) 
        else:
            signal_powers = jnp.abs(signal_powers[0])
        powers.append(signal_powers)
        noises.append(result[2])

    return sensitivity_function(noises, powers)


def sensitivity_q_noise(noises, powers):
    return jnp.abs(noises[0] / powers[0])


def sensitivity_qamplfreq_noise(noises, powers, frequencies):
    q_noise = jnp.abs(noises[0])
    ampl_noise = powers[1] * 4e-9
    freq_noise = (powers[2].T * frequencies).T * 1e-8
    return jnp.sqrt(q_noise**2 + ampl_noise**2 + freq_noise**2) / powers[0]


def evaluate_setups(
        setups,  
        frequencies, 
        bounding_function,
        calculate_loss,
        sensitivity_function,
        folder = None,
        suffix = "",
        reference_sensitivities = None,
        parameters = None,
        optimization_pairs = None,
        bounds = None, 
        homodyne=True,
    ): 
    """
    Evaluates given setups and plots different information about them.
    """
    if parameters is not None:
        for setup in setups:
            update_setup(parameters, optimization_pairs, bounds, setup, bounding_function)

    simulation_results = run_setups(setups, frequencies)
    sensitivities = calculate_sensitivities(simulation_results, sensitivity_function, homodyne=homodyne)
    powers = calculate_powers(simulation_results[0][0], *simulation_results[0][3:])

    if reference_sensitivities is None:
        reference_sensitivities = sensitivities

    sensitivity_loss, penalty, violations = calculate_loss(sensitivities, reference_sensitivities, powers)
    loss = float(sensitivity_loss + penalty)

    if folder is not None:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        plot_powers(*powers, suffix, folder)
        plot_comparison(frequencies,  
                        sensitivities,
                        reference_sensitivities,
                        folder,  
                        name=f"sensitivity{suffix}")
    
    penalty_data = {}
    penalty_data['loss'] = loss
    penalty_data['sensitivity_loss'] = float(sensitivity_loss)
    penalty_data['violations'] = violations.tolist()
    penalty_data['penalty'] = float(penalty)
    penalty_data['hard_side_powers'] = powers[0].tolist()
    penalty_data['soft_side_powers'] = powers[1].tolist()
    penalty_data['detector_powers'] = powers[2].tolist()
    penalty_data['violating'] = bool(jnp.any(violations > 0))

    return sensitivities, loss, penalty_data, setups[0]


def get_initial_guess(
        component_parameter_pairs,
        setups,
        frequencies,
        bounds,
        reference_sensitivities,
        bounding_function,
        calculate_loss,
        sensitivity_function,
        pool_size = 100,
        random_seed = None
    ):
    """
    Evaluates given setups with different sets of parameters.
    """
    rng = np.random.default_rng(random_seed)
    guesses = jnp.array(rng.uniform(-10, 10, (pool_size, len(component_parameter_pairs))))

    simulation_results = run_setups_with_parameter_sets(
        setups,
        frequencies,
        guesses,
        bounds,
        component_parameter_pairs,
        bounding_function
    )

    homodyne = False
    for node in setups[0].nodes:
        if node[1]["component"] == "qhd":
            homodyne = True

    sensitivities = calculate_sensitivities(simulation_results, sensitivity_function, homodyne=homodyne)
    powers = calculate_powers(simulation_results[0][0], *simulation_results[0][3:])

    losses, penalties, _ = calculate_loss(sensitivities, reference_sensitivities, powers)
    losses = losses + penalties

    return guesses[jnp.argmin(losses)], losses
