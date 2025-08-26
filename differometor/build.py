import numpy as np
import jax.numpy as jnp
from differometor.setups import Setup
from collections import defaultdict
from differometor.components import (
    F, DEFAULT_REFRACTIVE_INDEX, DEFAULT_PROPERTIES, FUNCTIONS, 
    signal_function, 
    laser_np, 
    laser, 
    vacuum_quantum_noise, 
    squeezer,
    nothing_matrix,
    directional_beamsplitter_matrix,
    beamsplitter_matrix,
    mirror_matrix,
    space_modulation,
    space_lower,
    space_modulation_lower,
    space,
    laser_amplitude_modulation,
    laser_frequency_modulation,
    laser_frequency_modulation_lower,
    susceptibility,
    force_calculation_left,
    force_calculation_right,
    surface,
    loss_quantum_noise,
    corrected_optomechanical_phase_left,
    corrected_optomechanical_phase_right,
    dummy_function
)


LINKING_FUNCTIONS = []


def set_instructions(
        instructions, 
        node, 
        function_input_indices,  
        output_column_indices, 
        system_matrix_row_indices, 
        system_matrix_column_indices, 
        function_indices,
        output_row_indices=None,
        carrier_indices=None,
        carrier_row_indices=None,
        carrier_column_indices=None,
        system_size_for_sidebands=None
    ):  
    function_input_indices_shape = 7
    if function_input_indices.shape[1] < function_input_indices_shape:
        # add placeholder values
        function_input_indices = np.concatenate([function_input_indices, np.zeros((function_input_indices.shape[0], function_input_indices_shape - function_input_indices.shape[1]))], axis=1)

    assert function_input_indices.shape[1] == function_input_indices_shape, f"Function input indices have to be standardized to length {function_input_indices_shape}."
    assert len(output_column_indices) == len(system_matrix_row_indices), "Output column indices have to match the length of system matrix row indices."
    
    # often these are partially the same as system matrix indices, so make them
    # independent copies to avoid side effects from updates in the system matrix indices
    if carrier_indices is not None:
        carrier_indices = carrier_indices.copy()
        carrier_row_indices = carrier_row_indices.copy()
        carrier_column_indices = carrier_column_indices.copy()
    
    if system_size_for_sidebands is not None:
        # We have two sidebands in signal and noise matrices. It looks like this:
        # 
        # sideband_1,    0,          signal
        # 0,            sideband_2,  signal
        # signal,       signal,      signal
        # 
        # All system matrix indices that are below system_size in either row or column must get 
        # duplicated. While doing that we also have to update the corresponding output indices and
        # the carrier indices. 

        # There are entries which need to be updated, but the column stays the same (e.g. signal connectors)
        only_row_mask = (system_matrix_row_indices < system_size_for_sidebands) & (system_matrix_column_indices >= system_size_for_sidebands) 
        # There are entries which need to be updated, but the row stays the same (e.g. force entries for optomechanics)
        only_column_mask = (system_matrix_column_indices < system_size_for_sidebands) & (system_matrix_row_indices >= system_size_for_sidebands) 
        # There are entries where both row and column need to be updated (e.g. mirror entries)
        row_and_column_mask = (system_matrix_row_indices < system_size_for_sidebands) & (system_matrix_column_indices < system_size_for_sidebands)
        # pure signal entries still need to be shifted
        pure_signal_mask = (system_matrix_row_indices >= system_size_for_sidebands) & (system_matrix_column_indices >= system_size_for_sidebands)

        # as carrier part of signal matrix gets duplicated, signal entries need to be relocated
        system_matrix_column_indices[pure_signal_mask] += system_size_for_sidebands
        system_matrix_row_indices[pure_signal_mask] += system_size_for_sidebands
        system_matrix_column_indices[only_row_mask] += system_size_for_sidebands
        system_matrix_row_indices[only_column_mask] += system_size_for_sidebands

        new_system_matrix_row_indices = system_matrix_row_indices.copy()
        new_system_matrix_column_indices = system_matrix_column_indices.copy()
        new_output_column_indices = output_column_indices.copy()
        new_output_row_indices = output_row_indices.copy() if output_row_indices is not None else None

        if only_row_mask.any():
            # add new row entries for the second sideband (e.g. signal connectors)
            new_system_matrix_row_indices = np.concatenate([new_system_matrix_row_indices, system_matrix_row_indices[only_row_mask] + system_size_for_sidebands])
            # we still need to duplicate the column indices, but it stays the same column
            new_system_matrix_column_indices = np.concatenate([new_system_matrix_column_indices, system_matrix_column_indices[only_row_mask]])
            new_output_column_indices = np.concatenate([new_output_column_indices, output_column_indices[only_row_mask]])
            if output_row_indices is not None:
                new_output_row_indices = np.concatenate([new_output_row_indices, output_row_indices[only_row_mask]])

        if only_column_mask.any():
            # add new column entries for the second sideband (e.g. force entries for optomechanics)
            new_system_matrix_column_indices = np.concatenate([new_system_matrix_column_indices, system_matrix_column_indices[only_column_mask] + system_size_for_sidebands])
            new_system_matrix_row_indices = np.concatenate([new_system_matrix_row_indices, system_matrix_row_indices[only_column_mask]])
            new_output_column_indices = np.concatenate([new_output_column_indices, output_column_indices[only_column_mask]])
            if output_row_indices is not None:
                new_output_row_indices = np.concatenate([new_output_row_indices, output_row_indices[only_column_mask]])

        if row_and_column_mask.any():
            # add new column and row entries for the second sideband (e.g. mirror entries)
            new_system_matrix_column_indices = np.concatenate([new_system_matrix_column_indices, system_matrix_column_indices[row_and_column_mask] + system_size_for_sidebands])
            new_system_matrix_row_indices = np.concatenate([new_system_matrix_row_indices, system_matrix_row_indices[row_and_column_mask] + system_size_for_sidebands])
            new_output_column_indices = np.concatenate([new_output_column_indices, output_column_indices[row_and_column_mask]])
            if output_row_indices is not None:
                new_output_row_indices = np.concatenate([new_output_row_indices, output_row_indices[row_and_column_mask]])

        output_column_indices = new_output_column_indices
        output_row_indices = new_output_row_indices
        system_matrix_row_indices = new_system_matrix_row_indices
        system_matrix_column_indices = new_system_matrix_column_indices

        # do the same for carrier_indices
        if carrier_indices is not None:
            only_row_mask = (carrier_row_indices < system_size_for_sidebands) & (carrier_column_indices >= system_size_for_sidebands)
            only_column_mask = (carrier_column_indices < system_size_for_sidebands) & (carrier_row_indices >= system_size_for_sidebands)
            row_and_column_mask = (carrier_row_indices < system_size_for_sidebands) & (carrier_column_indices < system_size_for_sidebands)
            pure_signal_mask = (carrier_row_indices >= system_size_for_sidebands) & (carrier_column_indices >= system_size_for_sidebands)

            carrier_column_indices[pure_signal_mask] += system_size_for_sidebands
            carrier_row_indices[pure_signal_mask] += system_size_for_sidebands
            carrier_column_indices[only_row_mask] += system_size_for_sidebands
            carrier_row_indices[only_column_mask] += system_size_for_sidebands

            new_carrier_row_indices = carrier_row_indices.copy()
            new_carrier_column_indices = carrier_column_indices.copy()
            new_carrier_indices = carrier_indices.copy()

            if only_row_mask.any():
                new_carrier_row_indices = np.concatenate([new_carrier_row_indices, carrier_row_indices[only_row_mask] + system_size_for_sidebands])
                new_carrier_column_indices = np.concatenate([new_carrier_column_indices, carrier_column_indices[only_row_mask]])
                new_carrier_indices = np.concatenate([new_carrier_indices, carrier_indices[only_row_mask]])

            if only_column_mask.any():
                new_carrier_column_indices = np.concatenate([new_carrier_column_indices, carrier_column_indices[only_column_mask] + system_size_for_sidebands])
                new_carrier_row_indices = np.concatenate([new_carrier_row_indices, carrier_row_indices[only_column_mask]])
                new_carrier_indices = np.concatenate([new_carrier_indices, carrier_indices[only_column_mask]])

            if row_and_column_mask.any():
                new_carrier_column_indices = np.concatenate([new_carrier_column_indices, carrier_column_indices[row_and_column_mask] + system_size_for_sidebands])
                new_carrier_row_indices = np.concatenate([new_carrier_row_indices, carrier_row_indices[row_and_column_mask] + system_size_for_sidebands])
                new_carrier_indices = np.concatenate([new_carrier_indices, carrier_indices[row_and_column_mask]])

            carrier_indices = new_carrier_indices
            carrier_row_indices = new_carrier_row_indices
            carrier_column_indices = new_carrier_column_indices

    assert function_input_indices.shape[1] == function_input_indices_shape, f"Function input indices have to be standardized to length {function_input_indices_shape}."
    assert len(output_column_indices) == len(system_matrix_row_indices), "Output column indices have to match the length of system matrix row indices."

    instructions[node] = {
        # The indices of the parameters in the parameter vector that are necessary to calculate 
        # new values. The length of this always has to be standardized (3 in this case). In case
        # of multiple functions, this has to be an (N, 3) matrix.
        "function_input_indices": function_input_indices,
        # output_column_indices determines what outputs of the FUNCTION_INDICES function are 
        # used to update the system matrix. Indices here relate to the output of the function 
        # determined by the indices in "function_indices". The length of this array has to match 
        # the length of system_matrix_row_indices. In case of multiple functions, this is just an
        # array of N output indices which index the (N, 3) output matrix.
        "output_column_indices": output_column_indices,
        # system_matrix_indices determine where to place the outputs in the system matrix. Indices
        # here relate to the system matrix or signal matrix.
        "system_matrix_row_indices": system_matrix_row_indices,
        "system_matrix_column_indices": system_matrix_column_indices,
        # The index of the function used to process function_inputs and produce outputs that get
        # placed in the system matrix. Indices here relate to the FUNCTION_LIST in components.py
        # In case of multiple functions, this is just an array of function indices.
        "function_indices": function_indices
    }
    if output_row_indices is not None:
        instructions[node]["output_row_indices"] = output_row_indices
    # The following indices are necessary because we often need entries of the carrier solution to 
    # calculate entries of the signal system. As the entries to be calculated are potentially 
    # distributed over the signal matrix, we need to use indexing to select and distribute the carrier
    # solution entries accordingly. 
    if carrier_indices is not None:
        # indicates which entries of the solution of the carrier system are needed
        instructions[node]["carrier_indices"] = carrier_indices
        # indicates where the entries of the solution of the carrier system have to be multiplied into
        instructions[node]["carrier_row_indices"] = carrier_row_indices
        instructions[node]["carrier_column_indices"] = carrier_column_indices


def parameter_linking(
        new_parameters, 
        indices_to_link, 
        linked_names, 
        linking_function_indices, 
        parameters
    ):
    for new_parameter in new_parameters:
        if type(new_parameter) is not tuple:
            parameters.append(new_parameter)
        else:
            assert type(new_parameter[0]) is str, "Linked parameter name has to be a string."
            assert callable(new_parameter[1]), "Linked parameter value has to be callable."
            # dummy value for now
            parameters.append(0)
            # add the function to the list of functions to be called
            LINKING_FUNCTIONS.append(new_parameter[1])
            linking_function_indices.append(len(LINKING_FUNCTIONS) - 1)
            indices_to_link.append(len(parameters) - 1)
            linked_names.append(new_parameter[0])


def build(setup: Setup):
    """
    Takes a setup graph and returns the system matrix as well as instructions for the 
    parameter vector.
    """
    system_size = 0
    signal_size = 0
    matrix_positions = {}
    carrier_instructions = {}
    signal_instructions = {}
    # carrier frequency is always the first parameter, refractive index always the second
    # 0 is the default value for alpha for mirrors and qhd phases and is always the third parameter
    # This needs to be this way because these parameters are needed throughout the build 
    # process
    parameters = [F, DEFAULT_REFRACTIVE_INDEX, 0]
    parameter_names = ["", "", ""]
    signal_components = []
    space_to_signals = defaultdict(list)
    laser_to_signals = defaultdict(list)
    parameter_positions = {}
    noise_instructions = {}
    all_ports = set([])
    used_ports = set([])
    quantum_detector_number = 0
    free_masses = []
    signal_frequency_position = 0
    surfaces_to_refractive_index_parameter_position = defaultdict(dict)
    mirror_indices = []
    beamsplitter_indices = []
    isolator_indices = []
    indices_to_link = []
    linked_names = []
    linking_function_indices = []

    def load_defaults(data, component=None):
        """
        Loads default properties for missing properties.
        """
        if component is None:
            component = data["component"]
        if "properties" not in data:
            data["properties"] = DEFAULT_PROPERTIES[component].copy()
        else:
            default_dict = DEFAULT_PROPERTIES[component].copy()
            default_dict.update(data["properties"])
            data["properties"] = default_dict

    # First edge loop to calculate the carrier matrix size from spaces
    for (source, target, data) in setup.edges(data=True):
        load_defaults(data, "space")

        # define standard values for source and target port for edges
        if "source_port" not in data:
            data["source_port"] = "right"
        if "target_port" not in data:
            data["target_port"] = "left"

        source_node = setup.nodes[source]
        target_node = setup.nodes[target]        

        # add parameters of spaces here already, because they will be needed by surfaces as well
        parameter_positions[f"{source}_{target}"] = len(parameters)
        parameter_linking([data["properties"]["length"], data["properties"]["refractive_index"]],
                            indices_to_link,
                            linked_names,
                            linking_function_indices,
                            parameters)
        parameter_names += [f"{source}_{target}_length", f"{source}_{target}_refractive_index"]

        # here we save the refractive indices of the spaces at each surface.
        if source_node["component"] == "mirror":
            surfaces_to_refractive_index_parameter_position[source][data["source_port"]] = len(parameters) - 1
        if target_node["component"] == "mirror":
            surfaces_to_refractive_index_parameter_position[target][data["target_port"]] = len(parameters) - 1

        # A beamsplitter must have the same refractive
        # index on left & top and right & bottom respectively. See finesse > components > beamsplitter.py >
        # Beamsplitter > refractive_index_1 and refractive_index_2
        port_mapping = {
            "right": "right",
            "left": "left",
            "top": "left",
            "bottom": "right"
        }
        if source_node["component"] == "beamsplitter":
            port = port_mapping[data["source_port"]]
            if port in surfaces_to_refractive_index_parameter_position[source] and parameters[surfaces_to_refractive_index_parameter_position[source][port]] != parameters[-1]:
                raise ValueError("Beamsplitter has to have the same refractive index on left & top and right & bottom respectively.")
            surfaces_to_refractive_index_parameter_position[source][port] = len(parameters) - 1
        if target_node["component"] == "beamsplitter":
            port = port_mapping[data["target_port"]]
            if port in surfaces_to_refractive_index_parameter_position[target] and parameters[surfaces_to_refractive_index_parameter_position[target][port]] != parameters[-1]:
                raise ValueError("Beamsplitter has to have the same refractive index on left & top and right & bottom respectively.")
            surfaces_to_refractive_index_parameter_position[target][port] = len(parameters) - 1

        # The following two for loops only affect lasers and detectors that were connected
        # via space (edge) and without an explicit target.
        if source_node["component"] in ["laser", "squeezer"]:
            matrix_positions[f"{source}_{target}_source"] = system_size
            # space adds two rows to system matrix
            system_size += 2
            # collect ports to later filter unused ports
            all_ports.add(f"{source}_{target}.left")
            # Laser needs a target, so add that if laser was connected via space (edge)
            source_node["target"] = f"{source}_{target}_source"
        elif source_node["component"] in ["detector", "qnoised"]:
            raise ValueError("Detectors can only be targets of edges.")

        if target_node["component"] in ["detector", "qnoised"]:
            matrix_positions[f"{source}_{target}_target"] = system_size
            # space adds two rows to system matrix
            system_size += 2
            # collect ports to later filter unused ports
            all_ports.add(f"{source}_{target}.right")
            # Detector needs a target, so add that if detector was connected via space (edge)
            target_node["target"] = f"{source}_{target}_target"
            # Update default direction of in to out
            target_node["direction"] = "out"
        elif target_node["component"] in ["laser", "squeezer"]:
            raise ValueError("Lasers can only be sources of edges.")

    # First node loop to calculate the carrier matrix and signal sizes from 
    # components with submatrix and signal components
    for node, data in setup.nodes(data=True):
        load_defaults(data)

        # all components that have a submatrix
        if data["component"] in MATRIX_SIZES:
            # Calculate matrix dimensions and identify the position of the matrix components
            # (e.g. mirror, beamsplitter) along the diagonal of the system matrix
            matrix_size = MATRIX_SIZES[data["component"]]
            matrix_positions[node] = system_size
            if data["component"] == "mirror":
                surface_indices = mirror_indices
            elif data["component"] == "beamsplitter":
                surface_indices = beamsplitter_indices    
            elif data["component"] == "directional_beamsplitter":
                surface_indices = isolator_indices
            surface_indices.extend(range(system_size, system_size + matrix_size))
            system_size += matrix_size
            # collect ports to later filter unused ports
            if data["component"] in ["mirror"]:
                all_ports.add(node + '.left')
                all_ports.add(node + '.right')
            if data["component"] in ["beamsplitter"]:
                all_ports.add(node + '.top')
                all_ports.add(node + '.bottom')
                all_ports.add(node + '.left')
                all_ports.add(node + '.right')
        
        if data["component"] in ["laser", "squeezer", "detector", "qnoised"]:
            # define standard values for port and direction for lasers and detectors
            if "port" not in data:
                data["port"] = "left"
            if "direction" not in data:
                data["direction"] = "in"

        if data["component"] == "squeezer":
            # To be able to change the squeezing angle and db, we need to mark it as a signal component
            # which is used later in the pair_to_arrays function
            signal_components.append(node)
        
        if data["component"] in ["qnoised", "qhd"]:
            if not "auxiliary" in data or ("auxiliary" in data and not data["auxiliary"]):
                quantum_detector_number += 1

        if data["component"] == "signal":
            signal_components.append(node)
            matrix_positions[node] = signal_size
            # we cannot directly apply the modulations because we don't know where the target
            # components will end up in the system matrix. Thus we note the corresponding components
            # to apply the modulations later.
            try:
                target_component = setup.nodes[data["target"]]["component"]
                if target_component == 'laser':
                    laser_to_signals[data["target"]].append(node)
            except KeyError:
                # target is space, because spaces dont have a component key
                space_to_signals[data["target"]].append(node)
            signal_size += 1
        
        if data["component"] == "free_mass":
            signal_components.append(node)
            matrix_positions[node] = signal_size
            signal_size += 2
            free_masses.append(node)

        if data["component"] == "frequency":
            signal_components.append(node)
            # add signal frequency as parameter and note its position
            signal_frequency_position = len(parameters)
            parameters.append(data["properties"]["frequency"])
            parameter_names.append(f"{node}_frequency")

    detectors = {}
    # + 1 for the right hand side input vector
    carrier_matrix = np.zeros((system_size, system_size + 1), dtype=complex)
    eye = np.eye(system_size, dtype=complex)
    carrier_matrix[:, :system_size] = eye

    # this gets introduced to avoid conditionals in the JAX simulation part. If there is no signal we 
    # would have to introduce conditionals, but if we just add this dummy signal, we can avoid them.
    if signal_size == 0:
        signal_size = 1
    # signal matrix is system_size (carrier matrix size) + signal_size (number of signals) 
    # + 1 for rhs, * 2 for two sidebands
    # TODO: Maybe we should make the handling of the second sideband optional, as it is not always needed.
    signal_matrix = np.zeros((system_size * 2 + signal_size, system_size * 2 + signal_size + 1), dtype=complex)
    eye = np.eye(system_size * 2 + signal_size, dtype=complex)
    signal_matrix[:, :system_size * 2 + signal_size] = eye

    noise_matrix = np.zeros(signal_matrix[:, :-1].shape)
    noise_detectors = {}
    # refer to https://doi.org/10.5281/zenodo.821380 appendix D about quantum noise
    noise_selection_vectors = np.zeros((quantum_detector_number, 1, system_size * 2 + signal_size), dtype=complex)
    noise_detector_count = 0

    detector_indices = []
    # collect qhd phase parameter indices relative to parameter array
    qhd_parameter_indices = []
    # collect placing indices relative to noise_selection_vectors
    qhd_placing_indices = []

    # Identify indices of lasers and detectors in input and output vectors and fill the system matrix 
    # Second node loop necessary because detector and laser indices depend on the target component whose
    # position in the system matrix is only known after the first node loop.
    for node, data in setup.nodes(data=True):
        ### SIGNALS ###
        # Even so signals don't depend on the position of other components, we still need to know the final 
        # system size to set them. 
        if data["component"] == "signal":
            signal_index = matrix_positions[node]
            # This adds the signals parameters and sets indices for the rhs entry of the signal field
            parameter_linking([data["properties"]["amplitude"], data["properties"]["phase"]],
                                indices_to_link,
                                linked_names,
                                linking_function_indices,
                                parameters)
            parameter_names += [f"{node}_amplitude", f"{node}_phase"]
            set_instructions(signal_instructions, 
                             node, 
                             # amplitude, phase
                             function_input_indices=np.array([[len(parameters)-2, len(parameters)-1]]),
                             output_column_indices=np.array([0]),
                             # place it in the row of the signal (doesn't need to get duplicated for sidebands)
                             # * 2 because we need to take into account the second sideband and we don't do
                             # any index updating in set_instructions in this case
                             system_matrix_row_indices=np.array([system_size * 2 + signal_index]),
                             # signal right-hand-side is the last column of the signal matrix
                             system_matrix_column_indices=np.array([-1]),
                             function_indices=[FUNCTIONS.index(signal_function)])
        
        ### LASERS AND DETECTORS ###
        if data["component"] in ["laser", "detector", "qnoised", "squeezer"]:
            # target_index describes the position of the target to which the laser or detector is connected
            try:
                target_index = matrix_positions[data["target"]]
            except KeyError:
                # connected to space, so matrix instructions will only contain source or target index set
                # in first edge loop
                target_index = matrix_positions[data["target"] + ("_source" if data["port"] == "left" else "_target")]
            # port_offset is the offset of a port (e.g. input / output) within the submatrix of the target component
            try:
                port_offset = PORT_DICTS[setup.nodes[data["target"]]["component"]][data["port"]]
            except KeyError:
                # space (always 0 as there is only 1 port)
                port_offset = 0
            # direction_offset is the offset of an input / output within a port
            if data["direction"] not in ["in", "out"]:
                raise ValueError("Direction has to be either in or out.")
            
            target_is_matrix_component = False
            try:
                target_is_matrix_component = setup.nodes[data["target"]]["component"] in MATRIX_SIZES
            except KeyError:
                pass
            if not target_is_matrix_component:
                # In this case, laser or detector is connected to a space and the directions
                # are reversed to conform with Finesse.
                direction_offset = 1 if data["direction"] == "in" else 0
            else:
                # In this case, laser or detector is connected to a component and
                # input should come before output.
                direction_offset = 0 if data["direction"] == "in" else 1
                
            # component_index is the position of the laser or detector in the system matrix
            component_index = target_index + port_offset + direction_offset
            matrix_positions[node] = component_index
            if data["component"] == "detector":
                try:
                    sideband = data["sideband"]
                except KeyError:
                    sideband = "upper"
                if sideband == "lower":
                    component_index += system_size
                detectors[node] = component_index
                detector_indices.append(component_index)
            if data["component"] == "qnoised":
                if not "auxiliary" in data or ("auxiliary" in data and not data["auxiliary"]):
                    noise_detectors[node] = noise_detector_count
                    # Here we set the entry that gets multiplied with the carrier solution from the detector position
                    # see finesse > detectors > compute > quantum.pyx > QND0Workspace > fill_selection_vector()
                    # upper sideband
                    noise_selection_vectors[noise_detector_count, 0, component_index] = np.sqrt(2)
                    # lower sideband
                    noise_selection_vectors[noise_detector_count, 0, component_index + system_size] = np.sqrt(2)
                    noise_detector_count += 1
            if data["component"] == "laser":
                # Mark target port as used
                target_port = data["target"].replace('_source', '').replace('_target', '') + '.' + data["port"]
                used_ports.add(target_port)
                # fill right-hand-side with laser field
                carrier_matrix[component_index, system_size] = laser_np(**data["properties"])
                parameter_linking([data["properties"]["power"], data["properties"]["phase"]],
                                    indices_to_link,
                                    linked_names,
                                    linking_function_indices,
                                    parameters)
                parameter_names += [f"{node}_power", f"{node}_phase"]
                set_instructions(carrier_instructions, 
                                 node, 
                                 # power, phase
                                 function_input_indices=np.array([[len(parameters)-2, len(parameters)-1]]),
                                 output_column_indices=np.array([0]),
                                 system_matrix_row_indices=np.array([component_index]),
                                 # carrier right-hand-side is the last column of the carrier matrix
                                 system_matrix_column_indices=np.array([-1]),
                                 function_indices=[FUNCTIONS.index(laser)])
                # lasers are a source of quantum noise, so they get an entry in the noise matrix at their position
                set_instructions(noise_instructions,
                                 node,
                                 # placeholder
                                 function_input_indices=np.array([[0]]),
                                 output_column_indices=np.array([0]),
                                 system_matrix_row_indices=np.array([component_index]),
                                 system_matrix_column_indices=np.array([component_index]),
                                 function_indices=[FUNCTIONS.index(vacuum_quantum_noise)],
                                 system_size_for_sidebands=system_size)
            if data["component"] == "squeezer":
                # Mark target port as used
                target_port = data["target"].replace('_source', '').replace('_target', '') + '.' + data["port"]
                used_ports.add(target_port)
                parameter_linking([data["properties"]["db"], data["properties"]["angle"]],
                                    indices_to_link,
                                    linked_names,
                                    linking_function_indices,
                                    parameters)
                parameter_names += [f"{node}_db", f"{node}_angle"]
                set_instructions(noise_instructions,
                                 node,
                                 # db, angle
                                 function_input_indices=np.array([[len(parameters)-2, len(parameters)-1]]),
                                 output_column_indices=np.array([0, 1, 2, 3]),
                                 system_matrix_row_indices=np.array([component_index, component_index, component_index + system_size, component_index + system_size]),
                                 system_matrix_column_indices=np.array([component_index, component_index + system_size, component_index + system_size, component_index]),
                                 function_indices=[FUNCTIONS.index(squeezer)],)
        if data["component"] == "qhd":
            parameter_linking([data["properties"]["phase"]],
                                indices_to_link,
                                linked_names,
                                linking_function_indices,
                                parameters)
            parameter_names += [f"{node}_phase"]
            detector1_index = matrix_positions[data["detector1"]]
            detector2_index = matrix_positions[data["detector2"]]
            # the carrier field of the second detector gets rotated by the specified phase in upper and lower sideband
            qhd_parameter_indices.extend([len(parameters) - 1, len(parameters) - 1])
            qhd_placing_indices.extend([[noise_detector_count, 0, detector2_index],
                                       [noise_detector_count, 0, detector2_index + system_size]])
            # upper sideband
            noise_selection_vectors[noise_detector_count, 0, detector1_index] = np.sqrt(2)
            noise_selection_vectors[noise_detector_count, 0, detector2_index] = np.sqrt(2) 
            # lower sideband
            noise_selection_vectors[noise_detector_count, 0, detector1_index + system_size] = np.sqrt(2)
            noise_selection_vectors[noise_detector_count, 0, detector2_index + system_size] = np.sqrt(2) 
            noise_detector_count += 1

        ### NOTHING ###
        if data["component"] == "nothing":
            nothing_index = matrix_positions[node]
            matrix = nothing_matrix()
            # initialize the carrier matrix with the nothing matrix
            carrier_matrix[nothing_index:nothing_index + matrix.shape[0], nothing_index:nothing_index + matrix.shape[1]] = matrix

        ### DIRECTIONAL BEAMSPLITTER ###
        if data["component"] == "directional_beamsplitter":
            directional_beamsplitter_index = matrix_positions[node]
            matrix = directional_beamsplitter_matrix()
            carrier_matrix[directional_beamsplitter_index:directional_beamsplitter_index + matrix.shape[0], directional_beamsplitter_index:directional_beamsplitter_index + matrix.shape[1]] = matrix

        ### MIRRORS AND BEAMSPLITTERS ###
        if data["component"] in ["mirror", "beamsplitter"]:
            component_index = matrix_positions[node]
            parameter_positions[node] = len(parameters)
            if data["component"] == "mirror":
                refractive_index_left_position = surfaces_to_refractive_index_parameter_position[node].get("left", 1)
                refractive_index_right_position = surfaces_to_refractive_index_parameter_position[node].get("right", 1)
                matrix = mirror_matrix(data["properties"]["loss"], 
                                       data["properties"]["reflectivity"], 
                                       data["properties"]["tuning"], 
                                       F,
                                       parameters[refractive_index_left_position],
                                       parameters[refractive_index_right_position])
                parameter_linking([data["properties"]["loss"], data["properties"]["reflectivity"], data["properties"]["tuning"]],
                                    indices_to_link,
                                    linked_names,
                                    linking_function_indices,
                                    parameters)
                parameter_names += [f"{node}_loss", f"{node}_reflectivity", f"{node}_tuning"]
                loss_position = len(parameters) - 3
                # loss, reflectivity, tuning, carrier_frequency, 2 refractive indices, 2 is the parameter position of 0 as default for alpha
                function_input_indices = np.array([[len(parameters)-3, len(parameters)-2, len(parameters)-1, 0, refractive_index_left_position, refractive_index_right_position, 2]])
                # parameter position of frequency changes because of sideband
                signal_function_input_indices = np.array([[len(parameters)-3, len(parameters)-2, len(parameters)-1, signal_frequency_position, refractive_index_left_position, refractive_index_right_position, 2]])
                # These indices pick out reflectivity_entry, transmissivity_entry and reflectivity_entry_minus
                # in the order necessary for the mirror submatrix (indices are relative to the output of 
                # the reflectivity_transmissivity_tuning function in components.py)
                output_column_indices = np.array([0, 1, 1, 2])
                # These are now the indices of the matrix entries where the reflectivity and transmissivity
                # entries will be placed. See the mirror_matrix function in components.py to understand
                # the indices.
                system_matrix_row_indices = component_index + np.array([1, 1, 3, 3])
                system_matrix_column_indices = component_index + np.array([0, 2, 0, 2])
                noise_output_column_indices = np.array([0, 0])
                noise_system_matrix_indices = component_index + np.array([1, 3]) # output 1 and output 2
            elif data["component"] == "beamsplitter":
                refractive_index_left_position = surfaces_to_refractive_index_parameter_position[node].get("left", 1)
                refractive_index_right_position = surfaces_to_refractive_index_parameter_position[node].get("right", 1)
                matrix = beamsplitter_matrix(data["properties"]["loss"],
                                             data["properties"]["reflectivity"],
                                             data["properties"]["tuning"],
                                             F,
                                             parameters[refractive_index_left_position],
                                             parameters[refractive_index_right_position],
                                             data["properties"]["alpha"])
                parameter_linking([data["properties"]["loss"], data["properties"]["reflectivity"], data["properties"]["tuning"], data["properties"]["alpha"]],
                                    indices_to_link,
                                    linked_names,
                                    linking_function_indices,
                                    parameters)
                parameter_names += [f"{node}_loss", f"{node}_reflectivity", f"{node}_tuning", f"{node}_alpha"]
                loss_position = len(parameters) - 4
                # loss, reflectivity, tuning, carrier_frequency, 2 refractive indices, alpha
                function_input_indices = np.array([[len(parameters)-4, len(parameters)-3, len(parameters)-2, 0, refractive_index_left_position, refractive_index_right_position, len(parameters)-1]])
                # parameter position of frequency changes because of sideband
                signal_function_input_indices = np.array([[len(parameters)-4, len(parameters)-3, len(parameters)-2, signal_frequency_position, refractive_index_left_position, refractive_index_right_position, len(parameters)-1]])
                # These indices pick out reflectivity_entry, transmissivity_entry and reflectivity_entry_minus
                # in the order necessary for the beamsplitter submatrix (indices are relative to the output of 
                # the reflectivity_transmissivity_tuning function in components.py)
                output_column_indices = np.array([0, 1, 0, 1, 1, 2, 1, 2])
                # These are now the indices of the matrix entries where the reflectivity and transmissivity
                # entries will be placed. See the beamsplitter_matrix function in components.py to understand
                # the indices.
                system_matrix_row_indices = component_index + np.array([1, 1, 3, 3, 5, 5, 7, 7])
                system_matrix_column_indices = component_index + np.array([2, 4, 0, 6, 0, 6, 2, 4])
                noise_output_column_indices = np.array([0, 0, 0, 0])
                noise_system_matrix_indices = component_index + np.array([1, 3, 5, 7]) # outputs 1, 2, 3 and 4
            # initialize the carrier matrix with the mirror or beamsplitter matrix
            carrier_matrix[component_index:component_index + matrix.shape[0], component_index:component_index + matrix.shape[1]] = matrix
            set_instructions(carrier_instructions,
                             node,
                             function_input_indices=function_input_indices,
                             output_column_indices=output_column_indices,
                             system_matrix_row_indices=system_matrix_row_indices,
                             system_matrix_column_indices=system_matrix_column_indices,
                             function_indices=[FUNCTIONS.index(surface)])
            # Surface submatrix also has to change in signal run if e.g. tuning changes 
            set_instructions(signal_instructions,
                             node,
                             function_input_indices=signal_function_input_indices,
                             output_column_indices=output_column_indices,
                             system_matrix_row_indices=system_matrix_row_indices,
                             system_matrix_column_indices=system_matrix_column_indices,
                             function_indices=[FUNCTIONS.index(surface)],
                             system_size_for_sidebands=system_size)
            # losses are a source of quantum noise, so they get an entry in the noise matrix at their position
            set_instructions(noise_instructions,
                             node,
                             # loss
                             function_input_indices = np.array([[loss_position]]),
                             output_column_indices = noise_output_column_indices,
                             system_matrix_row_indices = noise_system_matrix_indices,
                             system_matrix_column_indices = noise_system_matrix_indices,
                             function_indices=[FUNCTIONS.index(loss_quantum_noise)],
                             system_size_for_sidebands=system_size)

    # Apply potential laser field modulations. This can only be done here, because both positions of laser
    # and signal need to be known, which were only fixed in the second node loop.
    for node in laser_to_signals:
        laser_index = matrix_positions[node]
        # update signal instructions with the corresponding system matrix row indices
        signal_matrix_indices = []
        function_indices = []
        for signal in laser_to_signals[node]:
            signal_index = matrix_positions[signal]
            # system_size * 2 for the two sidebands, then *2 for upper and lower sideband
            signal_matrix_indices += [system_size * 2 + signal_index] * 2
            target_property = setup.nodes[signal]["target_property"]
            if target_property == "amplitude":
                function_indices.extend([FUNCTIONS.index(laser_amplitude_modulation), FUNCTIONS.index(laser_amplitude_modulation)])
            elif target_property == "frequency":
                function_indices.extend([FUNCTIONS.index(laser_frequency_modulation), FUNCTIONS.index(laser_frequency_modulation_lower)])
            else:
                raise ValueError("Target property for laser modulation has to be either amplitude or frequency.")
        laser_signal_number = len(laser_to_signals[node])
        system_matrix_row_indices = np.array([laser_index, laser_index + system_size] * laser_signal_number)
        system_matrix_column_indices = np.array(signal_matrix_indices)
        # Here we set the instructions to fill signal connectors in the signal matrix. These 
        # are located in the column of the respective signal and in the row of the respective laser.
        set_instructions(signal_instructions,
                         node,
                         # frequency
                         function_input_indices=np.tile(np.array([[signal_frequency_position],
                                                                  [signal_frequency_position]]), (laser_signal_number, 1)),
                         # for each laser modulation we take the first output of the corresponding function
                         output_column_indices=np.array([0, 0] * laser_signal_number),
                         # e.g. for laser_signal_number = 1: [0, 1], for 2: [0, 1, 2, 3]
                         output_row_indices=np.arange(laser_signal_number * 2),
                         system_matrix_row_indices=system_matrix_row_indices,
                         system_matrix_column_indices=system_matrix_column_indices,
                         function_indices=function_indices,
                         carrier_indices=np.array([laser_index, laser_index] * laser_signal_number),
                         carrier_row_indices=system_matrix_row_indices,
                         carrier_column_indices=system_matrix_column_indices)

    # set entries for free masses. This can only be done here, as we need all parameters from 
    # the mirrors and beamsplitters.
    for node in free_masses:
        data = setup.nodes[node]
        try: 
            component_index = matrix_positions[data["target"]]
            parameter_position = parameter_positions[data["target"]]
            target_component = setup.nodes[data["target"]]["component"]
        except KeyError:
            raise ValueError("Free mass has to be connected to either a mirror or a beamsplitter.")
        if target_component == "mirror":
            # f is at index, z is at index + 1
            free_mass_index = matrix_positions[node] + system_size
            parameter_linking([data["properties"]["mass"]],
                                indices_to_link,
                                linked_names,
                                linking_function_indices,
                                parameters)
            parameter_names += [f"{node}_mass"]
            # f_to_z connector, 4 x mirror to force connector, 2 x z to mirror connector
            system_matrix_row_indices = np.array([free_mass_index + 1, free_mass_index, free_mass_index, free_mass_index, free_mass_index, component_index + 1, component_index + 3])
            system_matrix_column_indices = np.array([free_mass_index, component_index, component_index + 1, component_index + 2, component_index + 3, free_mass_index + 1, free_mass_index + 1])
            refractive_index_left_position = surfaces_to_refractive_index_parameter_position[data["target"]].get("left", 1)
            refractive_index_right_position = surfaces_to_refractive_index_parameter_position[data["target"]].get("right", 1)
            # fill coupling entries from f to z, fill coupling entries from mirrors to f, fill coupling entries from z to mirror outputs
            # See finesse > components > modal > mirror.pyx > single_z_mechanical_frequency_signal_calc to understand
            # why we need which entries. Also finesse > components > mechanical.pyx > FreeMass > fill
            set_instructions(signal_instructions, 
                             node, 
                             function_input_indices=np.array([# signal_frequency, mass, the zeros at the end are placeholder values which are not used.
                                                             [signal_frequency_position, len(parameters)-1, 0, 0, 0, 0, 0],
                                                             # 2 is position of 0 as default value for alpha, rest are placeholder values
                                                             [2, 0, 0, 0, 0, 0, 0],
                                                             # 2 is position of 0 as default value for alpha, refractive_index_left, refractive_index_right, rest are placeholder values
                                                             [2, refractive_index_left_position, refractive_index_right_position, 0, 0, 0, 0],
                                                             # signal_frequency, tuning, reflectivity, loss, refractive_index, 2 is the position of the default 0 for alpha set at the beginning, last 0 is placeholder value
                                                             [signal_frequency_position, parameter_position + 2, parameter_position + 1, parameter_position, refractive_index_left_position, 2, 0],
                                                             [signal_frequency_position, parameter_position + 2, parameter_position + 1, parameter_position, refractive_index_left_position, refractive_index_right_position, 2],
                                                             ]),
                             output_row_indices=np.array([0, 1, 1, 2, 2, 3, 4]),
                             output_column_indices=np.array([0] * 7),
                             system_matrix_row_indices=system_matrix_row_indices,
                             system_matrix_column_indices=system_matrix_column_indices,
                             function_indices=[FUNCTIONS.index(susceptibility),
                                               FUNCTIONS.index(force_calculation_left),
                                               FUNCTIONS.index(force_calculation_right),
                                               FUNCTIONS.index(corrected_optomechanical_phase_left),
                                               FUNCTIONS.index(corrected_optomechanical_phase_right)],
                            # 4 mirror fields, 2 mirror inputs
                            carrier_indices=np.array([component_index, component_index + 1, component_index + 2, component_index + 3, component_index, component_index + 2]),
                            carrier_row_indices=system_matrix_row_indices[1:],
                            carrier_column_indices=system_matrix_column_indices[1:],
                            system_size_for_sidebands=system_size)
        elif target_component == "beamsplitter":
            # f is at index, z is at index + 1
            free_mass_index = matrix_positions[node] + system_size
            # TODO: mass is not yet optimizable
            parameter_linking([data["properties"]["mass"]],
                                indices_to_link,
                                linked_names,
                                linking_function_indices,
                                parameters)
            parameter_names += [f"{node}_mass"]
            # f_to_z connector, 8 x beamsplitter to force connector, 4 x z to mirror connector (beamsplitter outputs)
            system_matrix_row_indices = np.array([free_mass_index + 1, free_mass_index, free_mass_index, free_mass_index, free_mass_index, free_mass_index, free_mass_index, free_mass_index, free_mass_index, component_index + 1, component_index + 3, component_index + 5, component_index + 7])
            system_matrix_column_indices = np.array([free_mass_index, component_index, component_index + 1, component_index + 2, component_index + 3, component_index + 4, component_index + 5, component_index + 6, component_index + 7, free_mass_index + 1, free_mass_index + 1, free_mass_index + 1, free_mass_index + 1])
            refractive_index_left_position = surfaces_to_refractive_index_parameter_position[data["target"]].get("left", 1)
            refractive_index_right_position = surfaces_to_refractive_index_parameter_position[data["target"]].get("right", 1)
            # fill coupling entries from f to z, fill coupling entries from mirrors to f, fill coupling entries from z to mirror outputs
            # See finesse > components > modal > beamsplitter.pyx > single_z_mechanical_frequency_signal_calc to understand
            # why we need which entries. Also finesse > components > mechanical.pyx > FreeMass > fill
            set_instructions(signal_instructions, 
                             node, 
                             function_input_indices=np.array([# signal_frequency, mass, the last two indices are placeholder values which are not used.
                                                             [signal_frequency_position, len(parameters)-1, 0, 0, 0, 0, 0],
                                                             # alpha, rest are placeholder values
                                                             [parameter_position + 3, 0, 0, 0, 0, 0, 0],
                                                             # alpha, refractive_index_left, refractive_index_right
                                                             [parameter_position + 3, refractive_index_left_position, refractive_index_right_position, 0, 0, 0, 0],
                                                             # signal_frequency, tuning, reflectivity, loss, refractive_index, alpha
                                                             [signal_frequency_position, parameter_position + 2, parameter_position + 1, parameter_position, refractive_index_left_position, parameter_position + 3, 0],
                                                             [signal_frequency_position, parameter_position + 2, parameter_position + 1, parameter_position, refractive_index_left_position, refractive_index_right_position, parameter_position + 3],
                                                             ]),
                             output_row_indices=np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4]),
                             output_column_indices=np.array([0] * 13),
                             system_matrix_row_indices=system_matrix_row_indices, 
                             system_matrix_column_indices=system_matrix_column_indices,
                             function_indices=[FUNCTIONS.index(susceptibility),
                                               FUNCTIONS.index(force_calculation_left),
                                               FUNCTIONS.index(force_calculation_right),
                                               FUNCTIONS.index(corrected_optomechanical_phase_left),
                                               FUNCTIONS.index(corrected_optomechanical_phase_right)],
                            # 8 beamsplitter fields, 4 beamsplitter inputs (top input for left output, left input for top output, bottom input for right output, right input for bottom output)
                            carrier_indices=np.array([component_index, component_index + 1, component_index + 2, component_index + 3, component_index + 4, component_index + 5, component_index + 6, component_index + 7, component_index + 2, component_index, component_index + 6, component_index + 4]),
                            carrier_row_indices=system_matrix_row_indices[1:],
                            carrier_column_indices=system_matrix_column_indices[1:],
                            system_size_for_sidebands=system_size)

    # Identify the indices of the connector entries in the system matrix and fill the system matrix
    for (source, target, data) in setup.edges(data=True):
        source_node = setup.nodes[source]
        target_node = setup.nodes[target]

        # Mark source and target ports as used
        if source_node["component"] in MATRIX_SIZES:
            used_ports.add(source + '.' + data["source_port"])
        if target_node["component"] in MATRIX_SIZES:
            used_ports.add(target + '.' + data["target_port"])

        source_index = matrix_positions[source]
        target_index = matrix_positions[target]
        try:
            source_port_index = PORT_DICTS[setup.nodes[source]["component"]][data['source_port']]
            source_input_index = source_index + source_port_index 
            source_output_index = source_index + source_port_index + 1
        except KeyError:
            # laser is at input index (See couplings.excalidraw)
            source_input_index = source_index - 1
            source_output_index = source_index
        try:
            target_port_index = PORT_DICTS[setup.nodes[target]["component"]][data['target_port']]
            target_input_index = target_index + target_port_index
            target_output_index = target_index + target_port_index + 1
        except KeyError:
            # detector is at output index (See couplings.excalidraw)
            target_input_index = target_index + 1
            target_output_index = target_index

        # We use the full space function in case F changes at some point
        space_entry = space(jnp.array([F, data["properties"]["length"], data["properties"]["refractive_index"]]))[0]
        # initialize carrier matrix with space entries
        carrier_matrix[source_input_index, target_output_index] = space_entry
        carrier_matrix[target_input_index, source_output_index] = space_entry

        # parameters have been added in first edge loop already as they are needed by surfaces
        parameter_position = parameter_positions[f"{source}_{target}"]
        set_instructions(carrier_instructions,
                         f"{source}_{target}",
                         # carrier frequency, length, refractive_index
                         function_input_indices=np.array([[0, parameter_position, parameter_position + 1]]),
                         output_column_indices=np.array([0, 0]),
                         system_matrix_row_indices=np.array([source_input_index, target_input_index]),
                         system_matrix_column_indices=np.array([target_output_index, source_output_index]),
                         function_indices=[FUNCTIONS.index(space)])

        # all spaces need to be updated in the signal run because the frequency changes for the sidebands.
        # Lower sideband is handled manually here because so far the negative frequency is only important
        # for these space entries.
        set_instructions(signal_instructions,
                        f"{source}_{target}",
                        # signal_freqency, length, refractive_index
                        function_input_indices=np.array([[signal_frequency_position, parameter_position, parameter_position + 1],
                                                         [signal_frequency_position, parameter_position, parameter_position + 1]]),
                        output_row_indices=np.array([0, 0, 1, 1]),
                        output_column_indices=np.array([0, 0, 0, 0]),
                        # 2 upper sideband entries, 2 lower sideband entries
                        system_matrix_row_indices=np.array([source_input_index, target_input_index, source_input_index + system_size, target_input_index + system_size]),
                        system_matrix_column_indices=np.array([target_output_index, source_output_index, target_output_index + system_size, source_output_index + system_size]),
                        function_indices=[FUNCTIONS.index(space), FUNCTIONS.index(space_lower)])

        # Only now we can set the signal instructions for the strain signals acting on the respective spaces 
        # because we didn't know where these spaces would end up in the system matrix before.
        if f"{source}_{target}" in space_to_signals:
            # update signal instructions with the corresponding system matrix row indices
            signal_matrix_indices = []
            for signal in space_to_signals[f"{source}_{target}"]:
                signal_index = matrix_positions[signal]
                # system_size * 2 for the two sidebands, then *4 because 2 for upper and two for lower sideband
                signal_matrix_indices += [system_size * 2 + signal_index] * 4 
            space_signal_number = len(space_to_signals[f"{source}_{target}"])
            system_matrix_row_indices=np.array([source_input_index, target_input_index, source_input_index + system_size, target_input_index + system_size] * space_signal_number)
            system_matrix_column_indices=np.array(signal_matrix_indices)
            # Here we set the instructions to fill signal connectors in the signal matrix. Signal connectors are 
            # located in the column of the respective signal and in the row of the respective space. 
            set_instructions(signal_instructions,
                            # different key than in the space_signal case, because we don't want to override 
                            # the space_signal case and the keys are not needed any more, so we can choose any
                            f"{source}_{target}_modulation",
                            # frequency, length, refractive index
                            function_input_indices=np.tile(np.array([[signal_frequency_position, parameter_position, parameter_position + 1],
                                                                     [signal_frequency_position, parameter_position, parameter_position + 1]]), (space_signal_number, 1)),
                            # 2 signal entries * 2 sidebands for each signal
                            output_column_indices=np.array([0, 0, 0, 0] * space_signal_number),
                            # e.g. for space_signal_number = 1: [0, 0, 1, 1], for 2: [0, 0, 1, 1, 2, 2, 3, 3], ...
                            output_row_indices=np.repeat(np.arange(space_signal_number * 2), 2),
                            # two signal connector entries for each signal
                            system_matrix_row_indices=system_matrix_row_indices,
                            system_matrix_column_indices=system_matrix_column_indices,
                            function_indices=[FUNCTIONS.index(space_modulation), FUNCTIONS.index(space_modulation_lower)] * space_signal_number,
                            carrier_indices=np.array([source_input_index, target_input_index, source_input_index, target_input_index] * space_signal_number),
                            carrier_row_indices=system_matrix_row_indices,
                            carrier_column_indices=system_matrix_column_indices)

    # insert carrier matrix into signal matrix for first sideband
    signal_matrix[:system_size, :system_size] = carrier_matrix[:, :system_size]
    # insert carrier matrix into signal matrix for second sideband
    signal_matrix[system_size:-signal_size, system_size:-signal_size-1] = carrier_matrix[:, :system_size]

    # filter out unused ports and add vacuum noise to the noise matrix
    for node_port in all_ports - used_ports:
        node, port = node_port.split('.')
        try:
            port_offset = PORT_DICTS[setup.nodes[node]["component"]][port]
        except KeyError:
            # space
            port_offset = 0
        node_index = matrix_positions[node]
        port_index = node_index + port_offset
        # unused ports are a source of quantum noise, so they get an entry in the noise matrix at their position
        set_instructions(noise_instructions,
                         node_port,
                         # placeholder
                         function_input_indices=np.array([[0]]),
                         output_column_indices=np.array([0]),
                         system_matrix_row_indices=np.array([port_index]),  
                         system_matrix_column_indices=np.array([port_index]),
                         function_indices=[FUNCTIONS.index(vacuum_quantum_noise)],
                         system_size_for_sidebands=system_size)
        
    # convert linked_names to indices
    linked_indices = [parameter_names.index(name) for name in linked_names]
    if len(linked_indices) == 0:
        linked_indices = [0]
        indices_to_link = [0]
        LINKING_FUNCTIONS.append(lambda x: x)
        linking_function_indices = [0]

    return (
        # instructions
        (carrier_instructions,
        signal_instructions,
        noise_instructions,
        signal_components,
        parameter_names),
        # matrices
        (jnp.array([parameters], dtype=complex),
        jnp.array(carrier_matrix), 
        jnp.array(signal_matrix),
        jnp.array(noise_matrix, dtype=complex),
        jnp.array(noise_selection_vectors, dtype=complex),
        jnp.array(qhd_parameter_indices, dtype=int),
        jnp.array(qhd_placing_indices, dtype=int),
        jnp.array(linked_indices, dtype=int),
        jnp.array(linking_function_indices, dtype=int),
        jnp.array(indices_to_link, dtype=int)),
        # metadata
        (jnp.array(detector_indices, dtype=int),
        jnp.array(mirror_indices, dtype=int),
        jnp.array(beamsplitter_indices, dtype=int),
        jnp.array(isolator_indices, dtype=int)),
    )


def pairs_to_arrays(
        parameter_instructions: dict,
        signal_instructions: dict,
        noise_instructions: dict,
        signal_components: list,
        parameter_names: list,
        optimization_pairs: list = None, 
        changing_pairs: list = None
    ):
    # this will be used by both carrier and signal solver
    optimized_parameter_indices = []
    # optimization_value_indices is introduced to allow for multiple optimized parameters that all use the same value
    optimization_value_indices = []

    carrier_components = []
    carrier_changing_parameter_indices = []
    carrier_arrays = defaultdict(list)

    signal_changing_parameter_indices = []
    signal_arrays = defaultdict(list)

    noise_components = []
    noise_arrays = defaultdict(list)

    if optimization_pairs is None:
        optimization_pairs = []
    if changing_pairs is None:
        changing_pairs = []

    signal_component_number = 0
    # if there is a signal, all spaces and signal fields have to be updated after the carrier has been solved.
    # This is why (different from the carrier components) we have to iterate through all of them.

    for instruction in signal_instructions.values():
        # frequency only has instructions to update optimized_parameter_indices and nothing else
        # as it gets updated together with all the other entries.
        try:
            instruction["function_indices"]
        except KeyError:
            continue
        signal_arrays["function_input_indices"].append(instruction["function_input_indices"])
        signal_arrays["function_indices"].extend(instruction["function_indices"])
        signal_arrays["output_column_indices"].append(instruction["output_column_indices"])
        if "output_row_indices" in instruction:
            signal_arrays["output_row_indices"].append(instruction["output_row_indices"].flatten() + signal_component_number)
            individual_rows = len(np.unique(instruction["output_row_indices"]))
            signal_component_number += individual_rows
        else:
            signal_arrays["output_row_indices"].append(np.ones(len(instruction["output_column_indices"])) * signal_component_number)
            signal_component_number += 1
        signal_arrays["system_matrix_row_indices"].append(instruction["system_matrix_row_indices"])
        signal_arrays["system_matrix_column_indices"].append(instruction["system_matrix_column_indices"])
        if "carrier_indices" in instruction:
            signal_arrays["carrier_indices"].append(instruction["carrier_indices"])
            signal_arrays["carrier_row_indices"].append(instruction["carrier_row_indices"])
            signal_arrays["carrier_column_indices"].append(instruction["carrier_column_indices"])

    def append_information(instructions, arrays, component, components=None):
        component_function_input_indices = instructions[component]["function_input_indices"]
        # each component has functions that act on certain parameters. If a component is already in the list,
        # then all its parameters already get processed by the respective functions.
        if component in components:
            return None
        else:
            components.append(component)
        arrays["function_input_indices"].append(component_function_input_indices)
        arrays["function_indices"].extend(instructions[component]["function_indices"])
        arrays["output_column_indices"].append(instructions[component]["output_column_indices"])
        # the number of components equals the number of function calls and therefore the number of rows in the output.
        # all the outputs are in the same row
        arrays["output_row_indices"].append(np.ones(len(instructions[component]["output_column_indices"])) * (len(components) - 1))
        arrays["system_matrix_row_indices"].append(instructions[component]["system_matrix_row_indices"])
        arrays["system_matrix_column_indices"].append(instructions[component]["system_matrix_column_indices"])

    # compile noise arrays
    for component, instruction in noise_instructions.items():
        append_information(noise_instructions, noise_arrays, component, noise_components)

    # if changing or optimized components are from signal solver, we only have to update the parameter indices
    # because all entries will be updated anyway.
    for ix, optimization_pair in enumerate(optimization_pairs):
        if not isinstance(optimization_pair[0], list):
            optimized_component, optimized_parameter = optimization_pair
            try:
                optimized_parameter_indices.append(parameter_names.index(f"{optimized_component}_{optimized_parameter}"))
                optimization_value_indices.append(ix)
            except ValueError:
                pass
        else:
            # allow for multiple optimized parameters that all use the same value
            for optimized_component, optimized_parameter in optimization_pair:
                try:
                    optimized_parameter_indices.append(parameter_names.index(f"{optimized_component}_{optimized_parameter}"))
                    optimization_value_indices.append(ix)
                except ValueError:
                    pass

        if optimized_component not in signal_components and optimized_component in parameter_instructions:
            append_information(parameter_instructions, carrier_arrays, optimized_component, carrier_components)

    # multiple changing parameters can be updated at the same time (e.g. loss and reflectivity)
    for changing_component, changing_parameter in changing_pairs:
        if changing_component in signal_components:
            signal_changing_parameter_indices.append(parameter_names.index(f"{changing_component}_{changing_parameter}"))
        else:
            carrier_changing_parameter_indices.append(parameter_names.index(f"{changing_component}_{changing_parameter}"))
            append_information(parameter_instructions, carrier_arrays, changing_component, carrier_components)

    def combine_arrays(arrays, signal: bool = False):
        if len(arrays["function_input_indices"]) == 0:
            # In case there are no carrier or signal matrix entries to change, these parameters get set to 
            # dummy values that replace the 1 at position [0, 0] of the respective matrix with 1 (do nothing).
            # function_input_indices, function_indices, output_indices, system_matrix_indices
            return jnp.array([[0, 0, 0]]), jnp.array([FUNCTIONS.index(dummy_function)]), jnp.array([[0], [0]]), jnp.array([[0], [0]])
        else:
            function_input_indices = np.vstack(arrays["function_input_indices"])
            output_indices = np.stack((np.concatenate(arrays["output_row_indices"]), np.concatenate(arrays["output_column_indices"])))
            system_matrix_indices = np.stack((np.concatenate(arrays["system_matrix_row_indices"]), np.concatenate(arrays["system_matrix_column_indices"])))
            if signal:
                # dummy values
                carrier_solution_indices = np.array([0])
                # place in the first row of the last column of the system matrix which
                # should be guaranteed to be empty as it is the rhs of the signal system
                # which can only have non-zero entries in the last rows where the signals 
                # live 
                carrier_solution_placing_indices = np.array([[0], [-1]])
                if "carrier_indices" in arrays:
                    carrier_solution_indices = np.concatenate(arrays["carrier_indices"]),
                    carrier_solution_placing_indices = np.stack((np.concatenate(arrays["carrier_row_indices"]), np.concatenate(arrays["carrier_column_indices"])))
                return (jnp.array(function_input_indices, dtype=int),
                        jnp.array(arrays["function_indices"], dtype=int),
                        jnp.array(output_indices, dtype=int),
                        jnp.array(system_matrix_indices, dtype=int),
                        jnp.array(carrier_solution_indices, dtype=int),
                        jnp.array(carrier_solution_placing_indices, dtype=int))
            return (jnp.array(function_input_indices, dtype=int), 
                    jnp.array(arrays["function_indices"], dtype=int), 
                    jnp.array(output_indices, dtype=int), 
                    jnp.array(system_matrix_indices, dtype=int))

    return (
        # arrays to prepare
        (jnp.array(optimized_parameter_indices, dtype=int),
        jnp.array(optimization_value_indices, dtype=int),
        jnp.array(carrier_changing_parameter_indices, dtype=int),
        jnp.array(signal_changing_parameter_indices, dtype=int)),
        # carrier_arrays
        combine_arrays(carrier_arrays),
        # signal_arrays
        combine_arrays(signal_arrays, signal=True),
        # noise_arrays
        combine_arrays(noise_arrays)
    )


def prepare_arrays(
        optimized_parameter_indices, 
        optimized_value_indices, 
        carrier_changing_parameter_indices,
        signal_changing_parameter_indices,
        carrier_changing_values,
        parameters
    ):
    optimized_parameters = None

    # setup dummy arrays in case any array is empty to still comply with the vectorization scheme later
    # parameters has shape (1, N) where N is the number of parameters
    if len(optimized_parameter_indices) == 0:
        optimized_parameter_indices = jnp.array([0])
        optimized_parameters = jnp.array(parameters[0][optimized_parameter_indices])
        optimized_value_indices = jnp.array([0])

    if carrier_changing_values is not None:
        if len(carrier_changing_values.shape) == 1:
            carrier_changing_values = carrier_changing_values.reshape(1, -1)
        signal_changing_values = carrier_changing_values.copy()

    if len(carrier_changing_parameter_indices) == 0:
        carrier_changing_parameter_indices = jnp.array([0])
        # values needs to have shape (V, R) with V as number of changing parameters 
        # and R as number of values
        carrier_changing_values = jnp.array([parameters[0][carrier_changing_parameter_indices]])
    if len(signal_changing_parameter_indices) == 0:
        signal_changing_parameter_indices = jnp.array([0])
        signal_changing_values = jnp.array([parameters[0][signal_changing_parameter_indices]])

    return (
        optimized_parameters, 
        optimized_parameter_indices, 
        optimized_value_indices, 
        carrier_changing_parameter_indices, 
        carrier_changing_values, 
        signal_changing_parameter_indices, 
        signal_changing_values
    )


### SUPPORT DICTIONARIES ###


# the size of the component matrices
MATRIX_SIZES = {
    'mirror': 4,
    'beamsplitter': 8,
    'nothing': 4,
    'directional_beamsplitter': 8,
}


# The indices of the ports in the component matrices, counter-clockwise, 
# alternating between input and output, starting at the left input
PORT_DICTS = {
    'mirror': {
        'left': 0, 
        'right': 2
        },
    'beamsplitter': {
            'left': 0,
            'right': 4,
            'bottom': 6,
            'top': 2
        },
    'nothing': {
        'left': 0,
        'right': 2
    },
    'directional_beamsplitter': {
        'left': 0,
        'right': 4,
        'bottom': 6,
        'top': 2
    }
}
