import jax.numpy as jnp
import numpy as np


EPSILON_0C = 1
C_LIGHT = 299792458.0
LAMBDA = 1064e-9
UNIT_VACUUM = 1
# see finesse > simulations > base.pyx > ModelSettings > set_lambda0()
F0 = C_LIGHT / LAMBDA
# offset frequency of laser
F = 0 
H_PLANCK = 6.62607015e-34
X_SCALE = 1e-09
DEFAULT_REFRACTIVE_INDEX = 1.0
SOFT_SIDE_POWER_THRESHOLD = 2e3
HARD_SIDE_POWER_THRESHOLD = 3.5e6
DETECTOR_POWER_THRESHOLD = 1e-2


def standardize_output(output: list):
    output_length = len(output)
    # output length must be the same length as maximum output length so that all functions
    # can be executed simultaneously and build one output matrix during the simulation
    if output_length < 4:
        output.extend([0] * (4 - output_length))
    return jnp.array(output, dtype=jnp.complex128)


# default properties for components
DEFAULT_PROPERTIES = {
    'frequency': {'frequency': 1},
    'laser': {'power': 1., 'phase': 0.},
    'squeezer': {'db': 0, 'angle': 90},
    'mirror': {'loss': 5e-6, 'reflectivity': 0.5, 'tuning': 0.},
    'beamsplitter': {'loss': 5e-6, 'reflectivity': 0.5, 'tuning': 0., 'alpha': 45.},
    'free_mass': {'mass': 40.},
    'signal': {'amplitude': 1., 'phase': 0.},
    'space': {'length': 0, 'refractive_index': 1.},
    'detector': {},
    'qnoised': {},
    'qhd': {'phase': 180},
    'nothing': {},
    'directional_beamsplitter': {},
}


PARAMETER_BOUNDS = {
    "db": [0.01, 20],
    "angle": [-180, 180],
    "power": [0.01, 200],
    "loss": [5e-6, 0.999],
    "tuning": [0, 90],
    "mass": [0.01, 200],
    "length": [1, 4000],
    "reflectivity": [0, 1],
    "transmissivity": [0, 1],
    "phase": [-180, 180],
    "alpha": [-180, 180],
}


### SPACE ###


def space(parameters: jnp.ndarray):
    """
    parameters[0] = frequency
    parameters[1] = length
    parameters[2] = refractive index
    
    See finesse > components > modal > space.pyx > c_space_signal_fill > space_fill_optical_2_optical
    """
    phi = -jnp.exp(-1j * 2 * jnp.pi * parameters[0] * parameters[1] * parameters[2] / C_LIGHT)
    return standardize_output([phi])

def space_lower(parameters: jnp.ndarray):
    """
    parameters[0] = frequency
    parameters[1] = length
    parameters[2] = refractive index
    """
    parameters = jnp.array([-parameters[0], parameters[1], parameters[2]])
    return space(parameters)


### LASER ###


def laser(parameters: jnp.ndarray):
    """
    parameters[0] = power
    parameters[1] = phase
    """
    field = jnp.sqrt(2 * parameters[0] / EPSILON_0C) * jnp.exp(1j*jnp.radians(parameters[1]))
    return standardize_output([field])

def laser_np(power = 1., phase = 0.):
    return np.sqrt(2 * power / EPSILON_0C) * np.exp(1j*np.radians(phase))


### SQUEEZER ###


def squeezer(parameters: jnp.ndarray):
    """
    parameters[0] = db
    parameters[1] = angle
    
    See finesse > components > modal > squeezer.pyx > c_squeezer_fill_qnoise
    """
    vacuum_unit = UNIT_VACUUM / 2
    squeezing_parameter = parameters[0] / (20 * jnp.log10(jnp.e))
    phase = jnp.exp(1j * 2 * jnp.radians(parameters[1]))
    upper_qn_diagonal = vacuum_unit * jnp.cosh(2 * squeezing_parameter)
    upper_qn_off_diagonal = vacuum_unit * jnp.sinh(2 * squeezing_parameter) * phase
    lower_qn_diagonal = jnp.conjugate(upper_qn_diagonal)
    lower_qn_off_diagonal = jnp.conjugate(upper_qn_off_diagonal)
    return standardize_output([upper_qn_diagonal, upper_qn_off_diagonal, lower_qn_diagonal, lower_qn_off_diagonal])


### SIGNALS ###


def signal_function(parameters: jnp.ndarray):
    """
    parameters[0] = amplitude
    parameters[1] = phase
    """
    return standardize_output([parameters[0] * jnp.exp(1j*jnp.radians(parameters[1]))])

def laser_amplitude_modulation(parameters: jnp.ndarray):
    """
    See finesse > components > modal > laser.pyx > c_laser_fill_signal > SIGAMP_P1o

    This gets multiplied with the carrier solution during the simulation of the signal step.
    """
    factor = EPSILON_0C * 0.5
    return standardize_output([factor])

def laser_frequency_modulation(parameters: jnp.ndarray):
    """
    parameters[0] = signal frequency

    See finesse > components > modal > laser.pyx > c_laser_fill_signal > SIGFRQ_P1o

    This gets multiplied with the carrier solution during the simulation of the signal step.
    """
    factor = EPSILON_0C * 0.5 / parameters[0]
    return standardize_output([factor])

def laser_frequency_modulation_lower(parameters: jnp.ndarray):
    """
    parameters[0] = signal frequency
    """
    parameters = jnp.array([-parameters[0]])
    return laser_frequency_modulation(parameters)

def space_modulation(parameters: jnp.ndarray):
    """
    parameters[0] = signal frequency
    parameters[1] = length
    parameters[2] = refractive index

    See https://arxiv.org/abs/1306.6752 for more information.

    See finesse > components > modal > space.pyx > strain_signal_fill
    """
    # here w_g is actual frequency (no offset). It becomes offset later because sidebands have 
    # w_g as offset from carrier frequency (equation 6).
    w_g = 2 * jnp.pi * parameters[0]
    # this is the same as f0 in Finesse
    w_0 = 2 * jnp.pi * F0
    # equation 14 without the signal amplitude as this gets multiplied later
    m_g = - 0.5 * w_0 / w_g * jnp.sin(w_g * parameters[1] * parameters[2] / 2 / C_LIGHT)
    # equation 15 without the signal phase as this gets multiplied later
    phi_sb = - w_g * parameters[1] * parameters[2] / 2 / C_LIGHT
    # 1j for the pi/2 in equation 17, omega L/ c in equation 17 is zero because L is multiple of lambda
    z = 1j * m_g * jnp.exp(1j * phi_sb)
    return standardize_output([z])

def space_modulation_lower(parameters: jnp.ndarray):
    """
    parameters[0] = signal frequency
    parameters[1] = length
    parameters[2] = refractive index
    """
    parameters = jnp.array([-parameters[0], parameters[1], parameters[2]])
    return space_modulation(parameters)


### OPTOMECHANICS ###


def susceptibility(parameters: jnp.ndarray):
    """
    parameters[0] = signal frequency
    parameters[1] = mass

    See finesse > components > mechanical.pyx > FreeMass > fill
    """
    factor = -1 / (parameters[1] * (2 * jnp.pi * parameters[0]) ** 2)
    # TODO: minus was added because of discrepancy with Finesse. This should be checked.
    return - standardize_output([factor])

def force_calculation_left(parameters: jnp.ndarray):
    """
    parameters[0] = alpha (angle of incidence)

    See finesse > components > modal > mirror.pyx > c_mirror_signal_mech_fill, ws.field_to_F
    See finesse > components > modal > beamsplitter.pyx > c_beamsplitter_signal_fill, ws.field1_to_F
    """
    return - standardize_output([incidence_angle_left(parameters[0]) / (C_LIGHT * X_SCALE)])

def force_calculation_right(parameters: jnp.ndarray):
    """
    parameters[0] = alpha (angle of incidence)
    parameters[1] = refractive index left
    parameters[2] = refractive index right

    See finesse > components > modal > mirror.pyx > c_mirror_signal_mech_fill, ws.field_to_F
    See finesse > components > modal > beamsplitter.pyx > c_beamsplitter_signal_fill, ws.field2_to_F
    """
    return standardize_output([incidence_angle_right(parameters[0], parameters[1], parameters[2]) / (C_LIGHT * X_SCALE)])

def optomechanical_phase_left(alpha):
    """
    See finesse > components > modal > mirror.pyx > c_mirror_signal_mech_fill, ws.z_to_field
    See finesse > components > modal > beamsplitter.pyx > c_beamsplitter_signal_fill, ws.z_to_field1
    """
    # 2 * pi / lambda = k0
    return 1j * X_SCALE * 2 * jnp.pi / LAMBDA * incidence_angle_left(alpha)

def optomechanical_phase_right(alpha, refractive_index_left, refractive_index_right):
    """
    See finesse > components > modal > mirror.pyx > c_mirror_signal_mech_fill, ws.z_to_field
    See finesse > components > modal > beamsplitter.pyx > c_beamsplitter_signal_fill, ws.z_to_field2
    """
    # 2 * pi / lambda = k0
    return 1j * X_SCALE * 2 * jnp.pi / LAMBDA * incidence_angle_right(alpha, refractive_index_left, refractive_index_right)
    
def tuning_correction(parameters: jnp.ndarray):
    """
    parameters[0] = signal frequency
    parameters[1] = tuning

    See finesse > components > modal > mirror.pyx > single_z_mechanical_frequency_signal_calc
    """
    return jnp.exp(1j * jnp.radians(parameters[1]) * parameters[0] / F0)

def corrected_optomechanical_phase_left(parameters: jnp.ndarray):
    """
    parameters[0] = signal frequency
    parameters[1] = tuning
    parameters[2] = reflectivity
    parameters[3] = loss
    parameters[4] = refractive index left
    parameters[5] = alpha

    See finesse > components > modal > mirror.pyx > single_z_mechanical_frequency_signal_calc
    """
    absolute_reflectivity = (1 - parameters[3]) * parameters[2]

    factor = tuning_correction(parameters) * optomechanical_phase_left(parameters[5]) * reflectivity_left(absolute_reflectivity, parameters[1], parameters[0], parameters[4], parameters[5])
    return - standardize_output([factor])

def corrected_optomechanical_phase_right(parameters: jnp.ndarray):
    """
    parameters[0] = signal frequency
    parameters[1] = tuning
    parameters[2] = reflectivity
    parameters[3] = loss
    parameters[4] = refractive index left
    parameters[5] = refractive index right
    parameters[6] = alpha

    See finesse > components > modal > mirror.pyx > single_z_mechanical_frequency_signal_calc
    """
    absolute_reflectivity = (1 - parameters[3]) * parameters[2]

    # conjugated tuning correction and additional minus sign
    factor = jnp.conj(tuning_correction(parameters)) * -1 * optomechanical_phase_right(parameters[6], parameters[4], parameters[5]) * reflectivity_right(absolute_reflectivity, parameters[1], parameters[0], parameters[4], parameters[5], parameters[6])
    return - standardize_output([factor])


### QUANTUM NOISE ###


def vacuum_quantum_noise(parameters: jnp.ndarray):
    """
    See:
    - finesse > components > modal > laser.pyx > c_laser_fill_qnoise
    - finesse > components > workspace.pyx > c_optical_quantum_noise_plane_wave
    """
    # TODO: Understand role of frequency and add it? It is unclear how to handle multiple lasers
    # with different frequencies
    quantum_noise = UNIT_VACUUM / 2
    return standardize_output([quantum_noise])

def loss_quantum_noise(parameters: jnp.ndarray):
    """
    parameters[0] = loss

    See: 
    - finesse > components > modal > mirror.pyx > c_mirror_fill_qnoise
    - finesse > components > modal > beamsplitter.pyx > c_beamsplitter_fill_qnoise
    """
    # TODO: Understand role of frequency and add it? It is unclear how to handle multiple lasers
    # with different frequencies
    quantum_noise = parameters[0] / 2
    return standardize_output([quantum_noise])


### DETECTORS ###


def amplitude_detector(solution: jnp.ndarray):
    return jnp.sqrt(0.5 * EPSILON_0C)*jnp.abs(solution)

def power_detector(solution: jnp.ndarray):
    return 0.5 * EPSILON_0C * jnp.abs(solution) ** 2

def demodulate_signal_power(carrier: jnp.ndarray, signal: jnp.ndarray):
    """
    See finesse > detectors > compute > power.pyx > c_pd1_AC_output
    """
    upper_sideband = jnp.conj(carrier) * signal[:carrier.shape[0]]
    lower_sideband = carrier * signal[carrier.shape[0]:carrier.shape[0]*2]
    return upper_sideband + lower_sideband    


### MIRROR AND BEAMSPLITTER ###


"""
Parameters
----------
loss: float
    The loss of the mirror. Must be between 0 and 1.
reflectivity: float
    What fraction of 1 - loss is reflected. Must be between 0 and 1. This also 
    determines the transmissivity by (1 - loss) * (1 - reflectivity). 
    
    One problem for optimization are the constraints that come with reflectivity, 
    transmissivity and loss.

    1. Changing reflectivity, constant transmissivity:
        reflectivity can change in range 0-(1-transmissivity)
    2. Constant reflectivity, changing transmissivity:
        transmissivity can change in range 0-(1-reflectivity)
    3. Changing reflectivity, changing transmissivity:
        reflectivity can change in range 0-1
        transmissivity can change in range 0-(1-reflectivity)

    This is hard to implement in an optimization because two changing parameters 
    are dependent on each other.

    This could also be solved by optimizing two parameters in range 0-1. The first 
    parameter specifies the fraction that is taken up by the loss
    and the second parameter specifies that fraction of the previous fraction that
    is taken up by reflectivity. E.g. 0.5 and 0.2 means that loss is 0.5, and 
    reflectivity takes 20% of the other 50%, so 0.1 which means that transmissivity
    is at 0.4. Here the only constraint is that both parameters must be between 0 
    and 1.
"""


def surface(parameters: jnp.ndarray):
    """
    parameters[0] = loss
    parameters[1] = reflectivity
    parameters[2] = tuning
    parameters[3] = frequency
    parameters[4] = refractive_index_left
    parameters[5] = refractive_index_right
    parameters[6] = alpha
    """
    absolute_reflectivity = (1 - parameters[0]) * parameters[1]
    absolute_transmissivity = (1 - parameters[0]) * (1 - parameters[1])

    reflectivity_left_entry = reflectivity_left(absolute_reflectivity, parameters[2], parameters[3], parameters[4], parameters[6])
    transmissivity_entry = transmissivity(absolute_transmissivity, parameters[2], parameters[3], parameters[4], parameters[5], parameters[6])
    reflectivity_right_entry = reflectivity_right(absolute_reflectivity, parameters[2], parameters[3], parameters[4], parameters[5], parameters[6])

    return standardize_output([reflectivity_left_entry, transmissivity_entry, reflectivity_right_entry])


### MIRROR ###


def mirror_matrix(loss = 0., reflectivity = 0.5, tuning = 0., frequency = 0., refractive_index_left = 1., refractive_index_right = 1.):
    absolute_reflectivity = (1 - loss) * reflectivity
    absolute_transmissivity = (1 - loss) * (1 - reflectivity)

    reflectivity_left_to_left = reflectivity_left(absolute_reflectivity, tuning, frequency, refractive_index_left)
    transmissivity_entry = transmissivity(absolute_transmissivity, tuning, frequency, refractive_index_left, refractive_index_right)
    reflectivity_right_to_right = reflectivity_right(absolute_reflectivity, tuning, frequency, refractive_index_left, refractive_index_right)

    return np.array([
        [1, 0, 0, 0], # left input
        [reflectivity_left_to_left, 1, transmissivity_entry, 0], # left output
        [0, 0, 1, 0], # right input
        [transmissivity_entry, 0, reflectivity_right_to_right, 1] # right output
    ])


### BEAMSPLITTER ###


def beamsplitter_matrix(loss = 0., reflectivity = 0.5, tuning = 0., frequency = 0., refractive_index_left = 1., refractive_index_right = 1., alpha = 0.):
    absolute_reflectivity = (1 - loss) * reflectivity
    absolute_transmissivity = (1 - loss) * (1 - reflectivity)

    reflectivity_left_entry = reflectivity_left(absolute_reflectivity, tuning, frequency, refractive_index_left, alpha)
    transmissivity_entry = transmissivity(absolute_transmissivity, tuning, frequency, refractive_index_left, refractive_index_right, alpha)
    reflectivity_right_entry = reflectivity_right(absolute_reflectivity, tuning, frequency, refractive_index_left, refractive_index_right, alpha)

    return np.array([
        [1, 0, 0, 0, 0, 0, 0, 0], # left input
        [0, 1, reflectivity_left_entry, 0, transmissivity_entry, 0, 0, 0], # left output
        [0, 0, 1, 0, 0, 0, 0, 0], # top input
        [reflectivity_left_entry, 0, 0, 1, 0, 0, transmissivity_entry, 0], # top output
        [0, 0, 0, 0, 1, 0, 0, 0], # right input
        [transmissivity_entry, 0, 0, 0, 0, 1, reflectivity_right_entry, 0], # right output
        [0, 0, 0, 0, 0, 0, 1, 0], # bottom input
        [0, 0, transmissivity_entry, 0, reflectivity_right_entry, 0, 0, 1], # bottom output
    ])


### DIRECTIONAL BEAMSPLITTER ###


def directional_beamsplitter_matrix():
    """
    Left input -> Right output
    Top input -> Left output
    Right input -> Bottom output
    Bottom input -> Top output
    """
    return np.array([
        [1, 0, 0, 0, 0, 0, 0, 0], # left input
        [0, 1, -1, 0, 0, 0, 0, 0], # left output
        [0, 0, 1, 0, 0, 0, 0, 0], # top input
        [0, 0, 0, 1, 0, 0, -1, 0], # top output
        [0, 0, 0, 0, 1, 0, 0, 0], # right input
        [-1, 0, 0, 0, 0, 1, 0, 0], # right output
        [0, 0, 0, 0, 0, 0, 1, 0], # bottom input
        [0, 0, 0, 0, -1, 0, 0, 1] # bottom output
    ])


### NOTHING ###


def nothing_matrix():
    """
    Left input -> Right output
    Right input -> Left output
    """
    return np.array([
        [1, 0, 0, 0], # left input
        [0, 1, -1, 0], # left output
        [0, 0, 1, 0], # right input
        [-1, 0, 0, 1]  # right output
    ])


### HELPER FUNCTIONS ###


def incidence_angle_left(alpha):
    """
    alpha: angle of incidence in radians

    See finesse > components > modal > beamsplitter.pyx > BeamsplitterWorkspace
    """
    return jnp.cos(jnp.radians(alpha))

def incidence_angle_right(alpha, refractive_index_left, refractive_index_right):
    """
    alpha: angle of incidence in radians

    This calculates the beta angle based on the alpha angle using Snell's law. 
    See https://finesse.ifosim.org/docs/latest/physics/plane-waves/beam_splitter.html#beamsplitter-phase

    See finesse > components > modal > beamsplitter.pyx > BeamsplitterWorkspace
    """
    return jnp.cos(jnp.arcsin(refractive_index_left / refractive_index_right * jnp.sin(jnp.radians(alpha))))

def phase_shift(tuning, frequency, refractive_index):
    """
    See finesse > components > modal > mirror.pyx > mirror_fill_optical_2_optical
    See finesse > components > modal > beamsplitter.pyx > beamsplitter_fill_optical_2_optical
    """
    return 2 * jnp.radians(tuning) * refractive_index * (1 + frequency / F0)

def reflectivity_left(reflectivity, tuning, frequency, refractive_index_left, alpha = 0.):
    """
    See finesse > components > modal > mirror.pyx > mirror_fill_optical_2_optical
    See finesse > components > modal > beamsplitter.pyx > beamsplitter_fill_optical_2_optical
    """
    return -jnp.sqrt(reflectivity) * jnp.exp(1j * phase_shift(tuning, frequency, refractive_index_left) * incidence_angle_left(alpha))

def reflectivity_right(reflectivity, tuning, frequency, refractive_index_left, refractive_index_right = 1., alpha = 0.):
    """
    See finesse > components > modal > mirror.pyx > mirror_fill_optical_2_optical
    See finesse > components > modal > beamsplitter.pyx > beamsplitter_fill_optical_2_optical
    """
    return -jnp.sqrt(reflectivity) * jnp.exp(-1j * phase_shift(tuning, frequency, refractive_index_right) * incidence_angle_right(alpha, refractive_index_right, refractive_index_left))

def transmissivity(transmissivity, tuning, frequency, refractive_index_left, refractive_index_right, alpha = 0.):
    """
    See finesse > components > modal > mirror.pyx > mirror_fill_optical_2_optical
    See finesse > components > modal > beamsplitter.pyx > beamsplitter_fill_optical_2_optical
    """
    return -jnp.sqrt(transmissivity) * jnp.exp(1j * (jnp.pi / 2 + 0.5 * (phase_shift(tuning, frequency, refractive_index_left) * incidence_angle_left(alpha) - phase_shift(tuning, frequency, refractive_index_right) * incidence_angle_right(alpha, refractive_index_left, refractive_index_right))))


def dummy_function(parameters: jnp.ndarray):
    return standardize_output([1.])


FUNCTIONS = [
    force_calculation_right,
    laser, 
    surface, 
    space, 
    space_modulation, 
    signal_function, 
    dummy_function,
    vacuum_quantum_noise,
    loss_quantum_noise,
    space_lower,
    laser_amplitude_modulation,
    laser_frequency_modulation,
    susceptibility,
    force_calculation_left,
    corrected_optomechanical_phase_left,
    corrected_optomechanical_phase_right,
    space_modulation_lower,
    squeezer,
    laser_frequency_modulation_lower,
]
