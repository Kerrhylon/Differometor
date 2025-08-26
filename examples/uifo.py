import json
import differometor as df
import jax.numpy as jnp
import matplotlib.pyplot as plt
from differometor.setups import Setup
from differometor.setups import voyager
from differometor.components import demodulate_signal_power


### Calculate the Voyager sensitivity ###
#---------------------------------------#

print("Calculating Voyager sensitivity...")

# use a predefined Voyager setup with one noise detector and two signal detectors
S, _ = voyager()

# set the frequency range
frequencies = jnp.logspace(jnp.log10(800), jnp.log10(3000), 100)

# run the simulation with the frequency as the changing parameter
carrier, signal, noise, detector_ports, *_ = df.run(S, [("f", "frequency")], frequencies)

# calculate the signal power at the detector ports
powers = demodulate_signal_power(carrier, signal)
powers = powers[detector_ports]

# calculate the signal power from the two signal detectors for balanced homodyne detection
powers = powers[0] - powers[1]

# calculate the sensitivity
voyager_sensitivity = noise / jnp.abs(powers)

print("Voyager sensitivity calculation done!")


### Load pre-trained UIFO and calculate sensitivity ###
#-----------------------------------------------------#

print("Calculating pre-trained UIFO sensitivity...")

# Load the pre-trained UIFO model (no balanced homodyne detection, so only one detector)
with open("examples/data/uifo_800_3000.json", "r") as f:
    uifo = json.load(f)    

S = Setup.from_data(uifo)

# run the simulation with the frequency as the changing parameter
carrier, signal, noise, detector_ports, *_ = df.run(S, [("f", "frequency")], frequencies)

if len(detector_ports) == 1:
    # calculate the signal power at the detector port
    powers = demodulate_signal_power(carrier, signal)
    powers = powers[detector_ports].squeeze()
else:
    # calculate the signal power at the detector ports (balanced homodyne detection scheme)
    powers = demodulate_signal_power(carrier, signal)
    powers = powers[detector_ports]
    powers = powers[0] - powers[1]

# calculate the sensitivity
uifo_sensitivity = noise / jnp.abs(powers)

print("Pre-trained UIFO sensitivity calculation done!")


### Compare both sensitivity curves ###
#-------------------------------------#

plt.loglog(frequencies, voyager_sensitivity, label="Voyager")
plt.loglog(frequencies, uifo_sensitivity, label="UIFO")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Sensitivity [/sqrt(Hz)]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("output_uifo.png")
