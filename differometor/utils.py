import jax.numpy as jnp


def sigmoid_bounding(
        parameters, 
        bounds
    ):
    sigmoid_parameters = 1 / (1 + jnp.exp(-parameters))
    bounded_parameters = sigmoid_parameters * (bounds[1] - bounds[0]) + bounds[0]
    return bounded_parameters


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


def update_setup(
        parameters, 
        optimization_pairs, 
        bounds, 
        setup
    ):
    for ix, optimization_pair in enumerate(optimization_pairs):
        value = sigmoid_bounding(parameters[ix], bounds[:, ix])
        if isinstance(optimization_pair[0], list):
            for component_name, property_name in optimization_pair:
                set_value(component_name, property_name, value, setup)
        else:
            component_name, property_name = optimization_pair
            set_value(component_name, property_name, value, setup)
