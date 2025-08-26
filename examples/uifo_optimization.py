import os
import jax
import json
import optax
import copy
import numpy as np
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
from differometor.setups import voyager, uifo, constrain_inter_grid_cell_spaces
from differometor.utils import (
    sigmoid_bounding, 
    sensitivity_qamplfreq_noise, 
    calculate_sensitivities, 
    calculate_powers, 
    get_initial_guess,
    evaluate_setups,
)
from differometor.components import HARD_SIDE_POWER_THRESHOLD, SOFT_SIDE_POWER_THRESHOLD, DETECTOR_POWER_THRESHOLD
from differometor.simulate import run_setups, run_build_step, simulate_in_parallel


def calculate_loss(
        sensitivities, 
        reference_sensitivities, 
        powers,
    ):
    # calculate power violations (i.e. penalty based on much the power at each component exceeds its threshold)
    hard_side_violations = jnp.maximum(powers[0] / HARD_SIDE_POWER_THRESHOLD - 1, 0).squeeze(1)
    soft_side_violations = jnp.maximum(powers[1] / SOFT_SIDE_POWER_THRESHOLD - 1, 0).squeeze(1)
    detector_violations = jnp.maximum(powers[2] / DETECTOR_POWER_THRESHOLD - 1, 0).squeeze(1)

    violations = jnp.concatenate([
        hard_side_violations,
        detector_violations,
        soft_side_violations],
        axis=0
    )

    losses = jnp.mean(jnp.log10(sensitivities.T / reference_sensitivities), axis=-1)
    penalties = jnp.sum(violations.T, axis=-1)

    return losses, penalties, violations


folder = "output_uifo"
if not os.path.exists(folder):
    os.makedirs(folder, exist_ok=True)


### Calculate the target sensitivity ###
#--------------------------------------#

print("Calculating target sensitivity...")

# set the frequency range
frequencies = jnp.logspace(jnp.log10(20), jnp.log10(5000), 50)

# use a predefined Voyager setup with three different modulations (i.e. quantum noise, amplitude noise, frequency noise)
setups = [voyager(mode="space_modulation")[0], voyager(mode="amplitude_modulation")[0], voyager(mode="frequency_modulation")[0]]

# choose a sensitivity function that calculates sensitivities taking into account the three noise sources
sensitivity_function = partial(sensitivity_qamplfreq_noise, frequencies=frequencies)

# simulate the setups
simulation_results = run_setups(setups, frequencies)

# calculate the sensitivity values taking into account the three noise sources
reference_sensitivities = calculate_sensitivities(simulation_results, sensitivity_function)

# calculate the light power at all components within the setup
powers = calculate_powers(simulation_results[0][0], *simulation_results[0][3:])

# calculate the loss taking into account power violations
sensitivity_loss, penalty, _ = calculate_loss(reference_sensitivities, reference_sensitivities, powers)
reference_loss = float(sensitivity_loss + penalty)

print("Target sensitivity calculation done!")


### Start from random parameters and optimize the sensitivity ###
#---------------------------------------------------------------#

# select properties to be optimized
optimized_properties = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]

# specify the ranges for the properties to be optimized
property_bounds = {
    "db": [0, 10],
    "angle": [-360, 360],
    "power": [0, 200],
    "tuning": [-360, 360],
    "mass": [0.01, 200],
    "length": [0.1, 4000],
    "reflectivity": [0, 1],
}

# random seed for reproducability
random_seed = 42

# define a random uifo with three different modulations (i.e. quantum noise, amplitude noise, frequency noise)
q_noise_setup, component_property_pairs, centers, boundaries = uifo(size=3, 
                                                                    mode="space_modulation", 
                                                                    random=True, 
                                                                    verbose=True, 
                                                                    random_seed=random_seed)
ampl_noise_setup, _ = uifo(size=3, mode="amplitude_modulation", centers=centers, boundaries=boundaries)
freq_noise_setup, _ = uifo(size=3, mode="frequency_modulation", centers=centers, boundaries=boundaries)

# make sure the base setup never accidentially changes
q_noise_setup_function = lambda: copy.deepcopy(q_noise_setup) 
ampl_noise_setup_function = lambda: copy.deepcopy(ampl_noise_setup)
freq_noise_setup_function = lambda: copy.deepcopy(freq_noise_setup)
setups = lambda: [q_noise_setup_function(), ampl_noise_setup_function(), freq_noise_setup_function()]

# couple vertical and horizontal spaces at same positions, so that the grid structure of the uifo is always preserved
optimization_pairs = constrain_inter_grid_cell_spaces(component_property_pairs, optimized_properties)

# calculate the bounds for the properties to be optimized
lower_bounds = []
upper_bounds = []
for optimization_pair in optimization_pairs:
    if isinstance(optimization_pair[0], list):
        property_name = optimization_pair[0][1]
    else:
        property_name = optimization_pair[1]
    lower_bounds.append(property_bounds[property_name][0])
    upper_bounds.append(property_bounds[property_name][1])
bounds = np.array([lower_bounds, upper_bounds])

# initialize the uifo with 10 random sets of parameters and take the best parameter set as the initial guess
print("\nEvaluating different sets of initial parameters...\n")
initial_guess, losses = get_initial_guess(optimization_pairs, 
                                          setups(), 
                                          frequencies, 
                                          bounds, 
                                          reference_sensitivities,
                                          sigmoid_bounding,
                                          calculate_loss,
                                          sensitivity_function,
                                          pool_size=10,
                                          random_seed=random_seed)
print("\nInitial parameter evaluation done!")
print("Best initial guess index: ", jnp.argmin(losses))
print("Best initial guess loss: ", jnp.min(losses))

# check if the random uifo uses a balanced homodyne detection scheme
homodyne = False
for node in q_noise_setup_function().nodes:
    if node[1]["component"] == "qhd":
        homodyne = True

# build the three modulation setups
print("\nBuilding...")
q_arrays, *q_metadata = run_build_step(q_noise_setup_function(), [("f", "frequency")], frequencies, optimization_pairs)
ampl_arrays, *ampl_metadata = run_build_step(ampl_noise_setup_function(), [("f", "frequency")], frequencies, optimization_pairs)
freq_arrays, *freq_metadata = run_build_step(freq_noise_setup_function(), [("f", "frequency")], frequencies, optimization_pairs)
print("Building done!")


def objective_function(parameters):
    # map the parameters to between 0 and 1 and then to their respective bounds
    bounded_parameters = sigmoid_bounding(parameters, bounds)

    # simulate the three modulation setups
    q_results = simulate_in_parallel(bounded_parameters, *q_arrays[1:])
    ampl_results = simulate_in_parallel(bounded_parameters, *ampl_arrays[1:])
    freq_results = simulate_in_parallel(bounded_parameters, *freq_arrays[1:])
    results = [(*q_results, *q_metadata), (*ampl_results, *ampl_metadata), (*freq_results, *freq_metadata)]

    # calculate the sensitivities taking into account the three noise sources
    sensitivities = calculate_sensitivities(results, sensitivity_function, homodyne=homodyne)

    # calculate the light power at all components within the setup
    powers = calculate_powers(q_results[0], *q_metadata)

    # calculate the loss taking into account power violations
    sensitivity_loss, penalty, _ = calculate_loss(sensitivities, reference_sensitivities, powers)
    return sensitivity_loss + penalty


grad_fn = jax.jit(jax.value_and_grad(objective_function))
# warmup the function to compile it
print("Compiling...")
_ = grad_fn(initial_guess)
print("Compilation done!")

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=0.1)
)
optimizer_state = optimizer.init(initial_guess)

best_loss, best_params = 1e10, initial_guess
params, no_improve_count, losses = initial_guess, 0, []

print("\nOptimizing...\n")
for i in range(10):
    loss, grads = grad_fn(params)

    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss}")

    if loss < best_loss - 1e-4:
        best_loss, best_params, no_improve_count = loss, params, 0
        print(f"Iteration {i}: New best loss = {loss}")
    else:
        no_improve_count += 1

    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    params = optax.apply_updates(params, updates)
    losses.append(float(loss))

    # if the loss has not improved (< best_loss - 1e-4) over 1000 iterations, stop the optimization
    if no_improve_count > 1000:
        break
print("\nOptimization done!\n")

with open(f"{folder}/optimization_parameters.json", "w") as f:
    json.dump(best_params.tolist(), f, indent=4)

with open(f"{folder}/optimization_losses.json", "w") as f:
    json.dump(losses, f, indent=4)

plt.figure()
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.axhline(0, color="red", linestyle="--")
plt.grid()
plt.tight_layout()
plt.savefig(f"{folder}/optimization_losses.png")

print("Evaluating...")
(best_sensitivities, loss, penalty_data, best_setup) = evaluate_setups(setups(),
                                                                        frequencies,
                                                                        sigmoid_bounding,
                                                                        calculate_loss,
                                                                        sensitivity_function,
                                                                        folder,
                                                                        "_best",
                                                                        reference_sensitivities,
                                                                        best_params,
                                                                        optimization_pairs,
                                                                        bounds,
                                                                        homodyne=homodyne)

print(f"Evaluation done! You can find the results in the {folder} directory.")
print(f"Loss of best setup: {loss}")
