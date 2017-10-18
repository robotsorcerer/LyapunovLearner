### Learning Lyapunov Functions for Dynamical Systems [WIP]

This package contains the CLFDM algorithm presented in the paper:

**S.M. Khansari-Zadeh and A. Billard** (2014), "Learning Control Lyapunov Function
to Ensure Stability of Dynamical System-based Robot Reaching Motions." in
Robotics and Autonomous Systems, vol. 62, num 6, p. 752-765.

### Dependencies

All the python dependencies are pip installable. First, you would want an installation of python 3.5, and then `python-pip`. Install the remaining requirements by doing,

```bash
	pip install -r requirements.txt
```

from the root directory of this package.


### Code Structure

This package was ported from Khansari's [bitbucket](https://bitbucket.org/khansari/clfdm) matlab to python. It includes the python functions:
`demo.py`, `config.py` and 4 subdirectories: `clfm_lib`, `gmr_lib`, `example_models`, and `doc`.

+ [config.py](/config.py): a python script that configures the general properties of the Lyapunov energy function.

+ [demo.py](/demo.py): a python script illustrating how to use `clfm_lib` to learn an arbitrary model from a set of demonstrations. `clfm_lib` contains code which implements clfdm. See the [slides](/doc/SEDS_Slides.pdf) for further details about this library.

+ [gmr_lib](/gmr_lib): A library for Gaussian Mixture Model. This library is just for illustrative purposes. Feel free to use any library that you want for encoding the motion.

+ [example_models](/example_models) contains two handwriting motions recorded from a Tablet-PC.

+	[doc](/doc) includes the original [paper](/doc/Khansari_Billard_RAS2014.pdf) on CLFDM.

### Add Package to System Path

When running the demos, it is assumed that your current directory is the Lyapunov Learner directory. To allow module imports, we ask that you add the code directory to your PATH and PYTHON_PATH environment variables. In Linux, for example, this can be done by adding the following

```bash
export PATH=~/path/to/LyapunovLearner${PATH:+:${PATH}}
export PYTHONPATH=~/path/to/LyapunovLearner${PYTHONPATH:+:${PYTHONPATH}}
```

to your bashrc runfile.
