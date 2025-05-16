import differometor as df
from differometor.setups import aligo
import jax.numpy as jnp
from differometor.components import demodulate_signal_power
import matplotlib.pyplot as plt

# use a predefined aLIGO setup with one noise detector and one signal detector
S, _ = aligo()

# set the frequency range
frequencies = jnp.logspace(jnp.log10(20), jnp.log10(5000), 100)

# run the simulation with the frequency as the changing parameter
carrier, signal, noise, detector_ports, *_ = df.run(S, [("f", "frequency")], frequencies)

# calculate the signal power at the detector ports
powers = demodulate_signal_power(carrier, signal)
powers = powers[detector_ports[0]]

# calculate the sensitivity
sensitivity = noise / jnp.abs(powers)

plt.figure()
plt.loglog(frequencies, sensitivity)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Sensitivity [/sqrt(Hz)]")
plt.grid()
plt.tight_layout()
plt.savefig("output_aligo.png")
