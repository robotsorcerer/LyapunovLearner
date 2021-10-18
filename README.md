### One Hell of a Lyapunov Learner.

This code largely implements Learning CLFs using the SEDS paper by Khansari-Zadeh. See original code in [matlab](https://bitbucket.org/khansari/clfdm/src/master/demo_CLFDM_Learning.m).

+ Khansari-Zadeh's has a subtle example that illustrates the advantages of SEDS over DMPs, LWPRs, GMRs etc in his [2014 Autonomous Systems paper](/scripts/docs/AUS.pdf), and reproduced below:

<div align="center">
 <img src="/scripts/docs/seds_gmr.jpg" height="680px" width="600">
</div>

### Example Results on a Robot's Task Space Trajectories.

+ For the pre-recorded trajectories (demos) of the end-effector points in a 2D plane for the WAM robot (see the `*mat` files in the folder [scripts/data/](/scripts/data/)), if we run the [demo.py](/scripts/demo.py) file with the `w` model (resp. with the `s` model), we should have trajectory demos converging to a region of attractor at the origin  similar to  the left (resp. on the right) figure below

<div align="center">
 <img src="/scripts/docs/demos_w.jpg" height="400px" width="400px">
  <img src="/scripts/docs/demos_s.jpg" height="400px" width="400px">
</div>

Correcting the trajectories with Control Lyapunov Function, we will obtain the following:

<div align="center">
 <img src="/matlab/Doc/w_corrrections.gif" height="400px" width="400px">
  <img src="/matlab/Doc/s_corrrections.gif" height="400px" width="400px">
</div>

### Setup.

Dependencies:

+ scipy

+ numpy

+ matplotlib

All of these can be installed with

```
  pip install -r requirements.txt
```

And that about wraps up setting up!


### Usage

Basic Usage:

```
  python scripts/demo.py
```

Advanced Usage [with Options]:

  ```
    python scripts/demo.py [--silent|-si] [--model|-md] <s|w>  [--pause|-pz] <2> [--visualize|-vz] [--kappa0|-kp] <.1> [--rho0|-rh] <1.0>
  ```

  where angle brackets denote defaults.

#### Options

+ `--silent/-si`: Optimize the Lyapunov function in silent mode.

+ `--visualize/-vz`: Do we want to visualize the regions of attraction of the Lyapunov dynamics as we are updating the system on screen?

+ `--pause_time/-pz`: Time between updating the stabilization of the dynamical system on the pyplot display screen.

+ `--kappa0/-kp`: exponential coeff. in class-Kappa function.

+ `--rho0/-rh`: coeff. of class-Kappa function.

### FAQS

+ Why Consider this CLF correcction mechanism for stabilizing trajectories over Statistical Learning Methods or Dynamic Movement Primitives?

  -    Dynamic Movement Primitives are typically laden with disadvantages with learning multiple demos;

  -   Statistical Learning approaches, on the other hand, really do not have a guaranteed way of ensuring learned dynamics are Lyapunov stable;

  - Through a clever re-parameterization of robot trajectories, by so-called weighted sum of asymmetric quadratic functions (WSAQF), and nonlinear optimization, we learn stable attractors for the dynamics of a robot's motion, such that we are guaranteed to settle to correct attractors during optimization;

  - This code leverages a control Lyapunov function in deriving the control laws used to stabilize spurious learned trajectories;

+ This code is pretty much easy to follow and adapt for any dynamical system. Matter-of-factly, I used it in learning the dynamics of the Torobo 7-DOF arm in 2018 when I worked in Tokyo.

+ What is different between this implementation and Khansari-Zadeh's implementation?

  -  Well, for starters, a cleaner implementation of the Gaussian mixture models used in estimating dynamics along every trajectory sample.

### TODOs

+ Add quivers to level sets plot.

+ Add options for plotting different level sets for the Lyapunov Function.


### Citation

If you used `LyapunovLearner` in your work, please cite it:


```tex
@misc{LyapunovLearner,
  author = {Ogunmolu, Olalekan and Thompson, Rachel Skye and Pérez-Dattari, Rodrigo},
  title = {{Learning Control Lyapunov Functions}},
  year = {2021},
  howpublished = {\url{https://github.com/lakehanne/LyapunovLearner}},
  note = {Accessed February 10, 2020}
}
```
