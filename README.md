### One Hell of a Lyapunov Learner.

This code largely reimplements Learning CLFs using the SEDS paper by Khansari-Zadeh. See original code in [matlab here](https://bitbucket.org/khansari/clfdm/src/master/demo_CLFDM_Learning.m).

+ Khansari-Zadeh has a subtle example that illustrates the advantages of SEDS over DMPs, LWPRs, GMRs etc in his [2014 Autonomous Systems paper](scripts/docs/AUS.pdf), and reproduced below:

<div align="center">
 <img src="/scripts/docs/seds_gmr.jpg" height="540px" width="520">
</div>

### Example Results on a Robot's Task Space Trajectories.

+ For the pre-recorded trajectories of the end-effector pointsd in a 2D plane for the WAM robot (see the `*mat` files in the folder [scripts/data/](/scripts/data/)) if we run the [demo.py](/scripts/demo.py) file with the `w` model (resp. with the `s` model), you should obtain a chart with trajectories converging to the correct region of attractor  similar to what we have on the left (resp. on the right) below

<div align="center">
 <img src="/scripts/docs/energy_levels.png" height="333px" width="333px">
  <img src="/scripts/docs/energy_levels_sshape.png" height="333px" width="333px">
</div>


### Setup.

  Dependencies:

      + scipy

      + sklearn #>=0.19.0

      + numpy

      + matplotlib

  And that about wraps it up!


### Usage

Basic Usage:

```
  python scripts/demo.py
```

Usage with Options:

  ```
    python scripts/demo.py --silent --model s  -pz 2 --visualize
  ```

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
