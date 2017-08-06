### Learning Lyapunov Functions for Dynamical Systems

This package contains the CLFDM algorithm presented in the paper:

**S.M. Khansari-Zadeh and A. Billard** (2014), "Learning Control Lyapunov Function
to Ensure Stability of Dynamical System-based Robot Reaching Motions." 
Robotics and Autonomous Systems, vol. 62, num 6, p. 752-765.


### Code Structure

This code was ported from [bitbucket](https://bitbucket.org/khansari/clfdm) matlab package.

This package includes the python function: 
`demo.py`, and 4 subdirectories: `clfm_lib`, `gmr_lib`, `example_models`, and `doc`.


+ [demo.py](/demo.py): a python script illustrating how to use `clfm_lib` to learn an arbitrary model from a set of demonstrations. `clfm_lib` contains code which implements clfdm. See the [slides](/doc/SEDS_Slides.pdf) for further details about this library.

+ [gmr_lib](/gmr_lib): A library for Gaussian Mixture Model. This library is just for illustrative purposes. Feel free to use any library that you want for encoding the motion.

+ [example_models](/example_models) contains two handwriting motions recorded from a Tablet-PC.

+	[doc](/doc) includes the original [paper](/doc/Khansari_Billard_RAS2014.pdf) on CLFDM.

### Add Package to System Path

When running the demos, it is assumed that your current directory is the Lyapunov Learner directory. Otherwise, you should export the package to your system's path variable.