### One Hell of a Lyapunov Learner.

This code largely implements Learning CLFs using the SEDS paper by Khansari-Zadeh. See original code in [matlab](/matlab)

<!-- (https://bitbucket.org/khansari/clfdm/src/master/demo_CLFDM_Learning.m). -->

+ Khansari-Zadeh's has a subtle example that illustrates the advantages of SEDS over DMPs, LWPRs, GMRs etc in his [2014 Autonomous Systems paper](/scripts/docs/AUS.pdf), and reproduced below:

<div align="center">
 <img src="/scripts/docs/seds_gmr.jpg" height="680px" width="600">
</div>

### Learning Stable Task-Space Trajectories.

KZ recorded WAM robot end-effector trajectories on a 2D plane. The task is to stabilize pre-recorded trajectories using a combo of GMMs, Gaussian Mixture Regression, and Control Lyapunov Functions.

This code comes with two pre-recorded demos available in the data directory, i.e., [data](scripts/data). The main file is [demo.py](/scripts/demo.py) which executes the CLF corrected trajectories on the robot. The left image below denote a demonstrations of the robot drawing the letter `S` from three different initial conditions, and converging to an attractor at the origin; while the right image denote the Gaussian Mixture Regression-based CLF that corrects the trajectory in a piecewise manner as we feed the algothm the data.

#### S-Shaped Planar (Task-Space) Demos and Motion Corrections.

<div align="center">
 <img src="/scripts/docs/demos_s.jpg" height="400px" width="350px">
  <img src="/scripts/docs/corrected_traj_s.jpg" height="400px" width="350px">
</div>

<!-- #### S-Shaped Planar (Task-Space) Demos and Motion Corrections

<div align="center">
 <img src="/scripts/docs/demos_w.jpg" height="400px" width="350px">
  <img src="/scripts/docs/corrected_traj_w.jpg" height="400px" width="350px">
</div> -->

### Stable Trajectory Corrections

S-shaped Demo Corrections:

<div align="center">
 <img src="/scripts/docs/demos_s.jpg" height="400px" width="350px">
  <img src="/matlab/Doc/s_corrections.gif" height="400px" width="350px">
</div>

W-shaped Demo Corrections:

<div align="center">
 <img src="/scripts/docs/demos_w.jpg" height="400px" width="350px">
  <img src="/matlab/Doc/w_corrections.gif" height="400px" width="350px">
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

#### Basic Python Usage:

```
  python scripts/demo.py
```

#### Advanced Python Usage [with Options]:

  ```
    python scripts/demo.py [--silent|-si] [--model|-md] <s|w>  [--pause|-pz] <2> [--visualize|-vz] [--kappa0|-kp] <.1> [--rho0|-rh] <1.0> [--traj_num|-tn] <20e3>
  ```

  where angle brackets denote defaults.

#### Easy-peasy-lemon-squeezy advanced python usage:

  ```
    python scripts\demo.py -pz .1 --silent --model s -tn 20000                                               
  ```

#### Jupyter Notebook Interactive Example

Please find examples in the file [clf_demos.ipynb](/notes/clf_demos.ipynb).

#### Options

+ `--silent/-si`: Optimize the Lyapunov function in silent mode.

+ `--model/-md`: Which saved model to use? `w` or `s`.

+ `--visualize/-vz`: Do we want to visualize the regions of attraction of the Lyapunov dynamics as we are updating the system on screen?

+ `--pause_time/-pz`: Time between updating the stabilization of the dynamical system on the pyplot display screen.

+ `--kappa0/-kp`: Exponential coeff. in class-Kappa function.

+ `--traj_num/-tn`: Length of time to simulate trajectory corrections.

+ `--rho0/-rh`: Coeff. of class-Kappa function.

### FAQS

+ Why Consider this CLF correcction mechanism for stabilizing trajectories over Statistical Learning Methods or Dynamic Movement Primitives?

  -    Dynamic Movement Primitives are typically laden with disadvantages with learning multiple demos;

  -   Statistical Learning approaches, on the other hand, really do not have a guaranteed way of ensuring learned dynamics are Lyapunov stable;

  - Through a clever re-parameterization of robot trajectories, by so-called weighted sum of asymmetric quadratic functions (WSAQF), and nonlinear optimization, we learn stable attractors for the dynamics of a robot's motion, such that we are guaranteed to settle to correct attractors during optimization;

  - This code leverages a control Lyapunov function in deriving the control laws used to stabilize spurious learned trajectories;

+ This code is pretty much easy to follow and adapt for any dynamical system. Matter-of-factly, I used it in learning the dynamics of the Torobo 7-DOF arm in 2018 when I worked in Tokyo.

+ What is different between this implementation and Khansari-Zadeh's implementation?

  -  Well, for starters, a cleaner implementation of the Gaussian mixture models used in estimating dynamics along every trajectory sample.

  - A straightforward nonlinear optimization of the CLF cost

  - Better handling of constraints.

  - Written in python and easy to port to other open-source robot libraries.

### TODOs

+ Add quivers to level sets plot.

+ Add options for plotting different level sets for the Lyapunov Function.


### Citation

If you use `LyapunovLearner` in your work, please cite it:


```tex
@misc{LyapunovLearner,
  author = {Ogunmolu, Olalekan and Thompson, Rachel Skye and Pérez-Dattari, Rodrigo},
  title = {{Learning Control Lyapunov Functions}},
  year = {2021},
  howpublished = {\url{https://github.com/lakehanne/LyapunovLearner}},
  note = {Accessed February 10, 2020}
}
```
