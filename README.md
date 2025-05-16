# Differometor

[![PyPI version](https://img.shields.io/pypi/v/differometor)](https://pypi.org/project/differometor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[**Install guide**](#installation)
| [**Getting started**](#getting-started)
| [**Cavity example**](#cavity-example)
| [**Documentation**](#documentation)
| [**Development**](#development)
| [**Citing Differometor**](#citing-differometor)

**Differometor is a differentiable frequency domain interferometer simulator** implemented in Python using the [JAX](https://docs.jax.dev/en/latest/index.html) framework. Differometor’s implementation closely follows the established [Finesse simulator](https://finesse.ifosim.org/) and offers functionality to simulate plane waves in quasi-static, user-specified setup configurations including light field modulations, quantum noise calculations and optomechanical effects. 

Differometor is designed to become a powerful tool for state-of-the-art AI-driven design of novel fundamental physics experiments.

The main difference to existing interferometer simulators is the speed of interferometer optimizations. Running auto-differentiated interferometer optimizations with Differometor on a Nvidia Quadro RTX 6000 GPU can be up to 160 times faster than running numerically differentiated optimizations with Finesse on a Xeon E5-2698 v4 CPU. 

<img src="images/workflow.png" alt="workflow" width="1000"/>

Figure 1: Left: The Differentiable Interferometer Simulator (Differometor) takes a user-defined configuration (e.g., a 2×2 universal interferometer, UIFO) and efficiently computes its properties by making use
of just-in-time compilation, GPU acceleration, and automatic differentiation. This enables fast, gradient-based exploration of detector designs. Right: Gradient evaluation speedup over numerical approximation
with Finesse for various optical setups, as a function of optimized parameter count.

## Installation

This installation was tested on Ubuntu 24.04.1 LTS and on SUSE Linux Enterprise Server 15 SP6.

### From PyPI

```bash
pip install differometor
```

### Development mode

```bash
git clone https://github.com/artificial-scientist-lab/Differometor differometor
cd differometor
pip install -e .
```

### GPU support

Both installation methods above automatically install the CPU version of JAX. To upgrade to the GPU version with CUDA 12, please use:

```bash
pip install --upgrade "jax[cuda12]==0.5.0"
```


## Getting started

A good way to familiarize yourself with Differometor is to take a look at the [examples](examples):

   * [Optical cavity](examples/cavity.py): A simple case of modelling a Fabry-Pérot cavity. Compare this setup with the corresponding [Finesse example](https://finesse.ifosim.org/docs/latest/examples/01_simple_cav.html).
   * [Sensitivity of Advanced LIGO setup](examples/aligo.py): A more advanced example of computing the strain sensitivity of a simplified Advanced LIGO setup. Compare this setup with the corresponding [Finesse example](https://finesse.ifosim.org/docs/latest/examples/09_aligo_sensitivity.html).
   * [Sensitivity of Voyager setup](examples/voyager.py): Same as for the Advanced LIGO setup above, but with a Voyager configuration using balanced homodyne readout.
   * [Sensitivity optimization of Voyager setup](examples/voyager_optimization.py): This example showcases the potential of Differometor when it comes to interferometer optimization. It uses the Voyager setup from above, initializes its parameters randomly and then optimizes the parameters to achieve the best possible sensitivity.


## Cavity Example

[Optical cavity](examples/cavity.py): A simple case of modelling a Fabry-Pérot cavity. Compare this setup with the corresponding [Finesse example](https://finesse.ifosim.org/docs/latest/examples/01_simple_cav.html).

```python
import differometor as df
import jax.numpy as jnp
from differometor.components import power_detector
import matplotlib.pyplot as plt

# define a simple cavity setup with three detectors
S = df.Setup()
S.add("laser", "l0", power=1)
S.add("mirror", "m0", reflectivity=0.99, loss=0)
S.add("mirror", "m1", reflectivity=0.991, loss=0)
S.space("l0", "m0", length=1)
S.space("m0", "m1", length=1)
S.add("detector", "refl", target="m0", port="left", direction="out")
S.add("detector", "circ", target="m1", port="left", direction="in")
S.add("detector", "trns", target="m1", port="right", direction="out")

# set the tuning range
tunings = jnp.linspace(-180, 180, 400)

# run the simulation with the tuning as the changing parameter
carrier, signal, noise, detector_ports, *_ = df.run(S, [("m0", "tuning")], tunings)

# calculate the power
powers = power_detector(carrier)

# plot the power at the detector ports
plt.figure()
plt.plot(tunings, powers[detector_ports[0]], label="refl")
plt.plot(tunings, powers[detector_ports[1]], label="circ")
plt.plot(tunings, powers[detector_ports[2]], label="trns")
plt.yscale("log")
plt.xlabel("Tuning (degrees)")
plt.ylabel("Power (W)")
plt.grid()
plt.legend()
plt.show()
```

Output:

<img src="images/cavity.png" alt="cavity" width="500"/>

## Documentation

Differometor provides a Setup class to define the interferometer setup. You can add components and connect these components via spaces.

You can add the following components:

* **mirror**
    * ```setup.add("mirror", "m0")``` adds a mirror component with name m0 and default properties. Names cannot contain underscores. 
    * ```setup.add("mirror", "m1", reflectivity=0.5)``` adds a mirror component with name m1 which reflects half of the incoming light and transmits the other half. All other properties are set to default.
    * ```setup.add("mirror", "abc", reflectivity=0.25, loss=0.5)``` adds a mirror component with name abc which reflects 25% of the incoming light, looses 50% of the incoming light and transmits the other 25%.
    * ```setup.add("mirror", "m3", transmissivity=0.3, loss=0.6, tuning=90)``` adds a mirror component with name m3 which transmits 30% of the incoming light, looses 60% of the incoming light and is [tuned](https://finesse.ifosim.org/docs/latest/physics/plane-waves/beam_splitter.html#tuning) by 90 degrees.
    * The default properties of a mirror are:
        * reflectivity: 0.5
        * loss: 5e-6
        * tuning: 0
    * A mirror has two ports: left and right. Each port has two directions: in and out.
    * Reflectivity, transmissivity and loss always add up to 1 and are given as fractions of the total light amplitude. Differometor expresses reflectivity and transmissivity as one reflectivity parameter which is the fraction of the remaining light after loss that is reflected. Reflectivity and transmissivity of mirror m3 in the example above would therefore be expressed as the fraction: (1-0.3-0.6)/(1-0.6) = 0.25 
* **beamsplitter**
    * A beamsplitter is handled equivalent to a mirror, but has an additional alpha parameter describing the angle of incidence of the incoming light and two more ports.
    * ```setup.add("beamsplitter", "bs", alpha=50)``` adds a beamsplitter component with name bs which reflects half of the incoming light and transmits the other half at an angle of 50 degrees. All other properties are set to default.
    * A beamsplitter has four ports: left, right, top and bottom. Each port has two directions: in and out. Incoming light on the left port is transmitted to the right port and reflected to the top port. Incoming light at the top is transmitted to the bottom port and reflected to the left port. Incoming light at the right port is transmitted to the left port and reflected to the bottom port. Incoming light at the bottom port is transmitted to the top port and reflected to the right port.
    * The default properties of a beamsplitter are:
        * reflectivity: 0.5
        * loss: 5e-6
        * tuning: 0
        * alpha: 45
* **directional_beamsplitter**
    * ```setup.add("directional_beamsplitter", "dbs")``` adds a directional beamsplitter component with name dbs
    * A directional beamsplitter forwards all light at the left port to the right port, all light at the top port to the left port, all light at the right port to the bottom port and all light at the bottom port to the top port.
* **nothing**
    * ```setup.add("nothing", "n0")``` adds a nothing component with name n0. These have two ports, left and right. Each port has two directions: in and out. Nothing components can be used to connect components without using a space.
    * The left input goes to the right output and the right input goes to the left output. Nothing changes.
* **laser**
    * ```setup.add("laser", "l0")``` adds a laser component with name l0 and default properties.
    * ```setup.add("laser", "l1", power=0.5)``` adds a laser component with name l1 which has a power of 0.5. All other properties are set to default.
    * lasers can have a target if you want to connect them to a component without using a space. ```setup.add("laser", "l2", target="m0", port="left", direction="in")``` adds a laser component with name l2 which is directly connected to the left port of m0 in the in direction. All other properties are set to default.
    * The default properties of a laser are:
        * power: 1
        * phase: 0
* **squeezer**
    * ```setup.add("squeezer", "sq", db=10, angle=0)``` adds a squeezer with name sq producing a squeezed-light beam with given squeezing in decibels db and angle.
    * Like a laser above, a squeezer can also have an additional target, port and direction.
    * The default properties of a squeezer are:
        * db: 0
        * angle: 90
* **free_mass**
    * ```setup.add("free_mass", "sus", mass=10, target="m0")``` adds a simple free mass suspension with name sus and mass 10kg to the object with name m0. Masses can be added to mirrors or beamsplitters.
    * The default properties of a free mass are:
        * mass: 40
* **signal**
    * ```setup.add("signal", "s0", target="l0_amplitude")``` adds a amplitude signal generator with name s0 to the laser l0. 
    * ```setup.add("signal", "s1", target="abc_frequency")``` adds a frequency signal generator with name s1 to the laser abc.
    * ```setup.add("signal", "GW", target="m0_m1")``` adds a signal generator with name GW to the space between m0 and m1.
    * ```setup.add("signal", "s2", target="l0_amplitude", amplitude=("l0_power", jnp.sqrt))``` adds a signal generator with name s2 to the laser l0. The signal amplitude is defined as the square root of the laser power. So instead of actual values, properties can be defined as functions of other properties.
    * Signal generators can be connected only to spaces or lasers. For lasers, they can only be applied as amplitude (l0_amplitude) or frequency (l0_frequency) modulation. 
    * The default properties of a signal are:
        * amplitude: 1
        * phase: 0
* **qhd**
    * ```setup.add("qhd", "qhd0", detector1="noise1", detector2="noise2", phase=45)``` adds a balanced homodyne quantum noise detector with name qhd0 and a phase of 45 degrees. It uses the two noise detectors noise1 and noise2.
    * The default properties of a qhd are:
        * phase: 0
* **frequency**
    * ```setup.add("frequency", "f")``` sets the frequency used by all signals in the simulation to the default 1 Hz. It is a unique element and can only be set once in a given setup.
    * The default properties of a frequency are:
        * frequency: 1
* **detector**
    * ```setup.add("detector", "d0", target="bs", port="top", direction="out")``` adds a detector with name d0 to the top port of the object bs in the out direction.
    * ```setup.add("detector", "d1", target="m0")``` adds a detector with name d1 to the object m0. The default port and direction is left and in.
* **qnoised**
    * ```setup.add("qnoised", "1", target="m0")``` adds a noise detector with name 1 to the left port of the object m0 in the in direction. 
    * ```setup.add("qnoised", "noise", target="m0", port="left", direction="in", auxiliary=True)``` adds a noise detector with name noise to the left port of the object m0 in the in direction. This is an auxiliary detector that does not get its own noise output and can be used as detector1 or detector2 in a qhd.

You can connect components via spaces:

* **space**
    * ```setup.space("m0", "m1", length=0.5)``` adds a space between the objects m0 and m1 with a length of 0.5 meters. By default it connects the right port of m0 with the left port of m1.
    * ```setup.space("bs1", "bs2", length=100, source_port="top", target_port="bottom")``` adds a space between the objects bs1 and bs2 with a length of 100 meters. It connects the top port of bs1 with the bottom port of bs2.
    * ```setup.space("l0", "m0", length=1, refractive_index=1.5, target_port="right")``` adds a space between the laser l0 and the object m0 with a length of 1 meter and a refractive index of 1.5. It connects the laser l0 with the right port of m0, in the in direction.
    * The default properties of a space are:
        * length: 0
        * refractive_index: 1

For further details on the limits and inner workings of the simulator, please refer to the [Finesse documentation](https://finesse.ifosim.org/docs/latest/index.html).


## Development

Differometor is developed at the [Artificial Scientist Lab](https://mpl.mpg.de/research-at-mpl/independent-research-groups/krenn-research-group/) under Dr. Mario Krenn at the Max Planck Institute for the Science of Light in Erlangen, Germany.


## Citing Differometor

To cite this repository, please use the following BibTeX entry:

```
@software{differometor2025github,
  author = {Jonathan Klimesch and Yehonathan Drori and Rana X Adhikari and Mario Krenn},
  title = {Differometor: A Differentiable Interferometer Simulator for the Computational Design of Gravitational Wave Detectors},
  url = {http://github.com/artificial-scientist-lab/Differometor},
  version = {0.1.0},
  year = {2025},
}
```