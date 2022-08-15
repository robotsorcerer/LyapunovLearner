### One Hell of a Lyapunov Learner.

This code implements Learning Control Lyapunov Functions, or CLFs, by leveraging the _Stable Estimation of Dynamic Systems_ paper from Aude Billard's group at EPFL. For details, please consult

```
   S.M. Khansari-Zadeh and A. Billard (2014), "Learning Control Lyapunov Function
   to Ensure Stability of Dynamical System-based Robot Reaching Motions."
   Robotics and Autonomous Systems, vol. 62, num 6, p. 752-765.
```

One may find the MATLAB version of this implementation in the [matlab](/matlab) subfolder.

<!-- (https://bitbucket.org/khansari/clfdm/src/master/demo_CLFDM_Learning.m). -->

Khansari-Zadeh has a subtle example that illustrates the advantages of SEDS over DMPs, LWPRs, regress_gauss_mixs etc in his [2014 Autonomous Systems paper](/scripts/docs/AUS.pdf), and reproduced below:

<div align="center">
 <img src="/scripts/docs/seds_gmr.jpg" height="680px" width="600">
</div>

### Learning Stable Task-Space Trajectories.

For recorded WAM robot end-effector trajectories on a 2D plane (there are two pre-recorded demos available in the [data](scripts/data) directory), the task is to stabilize pre-recorded trajectories using a combo of GMMs, Gaussian Mixture Regression, and Control Lyapunov Functions -- all learned from data.

### Learning Stable Trajectories

S-shaped demos and reconstructions from three different initial conditions. The left image below denotes a demonstration of the robot drawing the letter `S` from three different initial conditions, and converging to an attractor at the origin; while the right image denotes the Gaussian Mixture Regression-based CLF that corrects the trajectories in a piecewise manner as we feed the algorithm the data.

<div align="center">
 <img src="/scripts/docs/demos_s.jpg" height="400px" width="320px">
 <img src="/scripts/docs/s_corrected_video.gif" height="440px" width="420px">
</div>

W-shaped demos from three different initial conditions.

<div align="center">
  <img src="/scripts/docs/demos_w.jpg" height="400px" width="350px">
  <img src="/scripts/docs/w_corrected.gif" height="400px" width="350px">
</div>

### Setup.

Dependencies: Scipy | Numpy | Matplotlib.

Install as follows:

  ```
    pip install -r requirements.txt .
  ```

And that about wraps up setting stuff up!


### Usage

#### Basic Python Usage:

```
  python scripts/demo.py
```

#### Python Usage [with options]:

  ```
    python scripts/demo.py [--silent|-si] [--model|-md] <s|w>  [--pause|-pz] <1e-4> [--visualize|-vz] [--kappa0|-kp] <.1> [--rho0|-rh] <1.0> [--traj_num|-tn] <20e3>
  ```

  where angle brackets denote defaults.

#### Options

+ `--silent/-si`: Optimize the control Lyapunov function in silent mode.

+ `--model/-md`: Which saved model to use? `w` or `s`.

+ `--visualize/-vz`: Do we want to visualize the regions of attractor (ROA) of the Lyapunov dynamics?

+ `--off_priors/-op`: Do we want to use the offline computed priors with KZ's Expectimax algo implementation?

+ `--pause_time/-pz`: Time between updating the stabilization of the dynamical system's reaching motions on the `pyplot` display screen.

+ `--kappa0/-kp`: Exponential coefficient in the class-Kappa function that guarantees `V` is positive outside the equilibrium point/region (see paper referenced above for details).

+ `--traj_num/-tn`: Length of time to simulate trajectory corrections.

+ `--rho0/-rh`: Coefficient of class-Kappa function (see paper referenced above for details).

#### Example python usage [with options]:

  ```
    python scripts\demo.py -pz .1 --silent --model s -tn 20000                                               
  ```

#### Jupyter Notebook Interactive Example

Please find the example in the file [clf_demos.ipynb](/notes/clf_demos.ipynb).


### FAQS

+ Why Gaussian and Weighted Regression and not Neural Networks

Gaussian mixtures and locally-weighted regression are efficient and accurate, as opposed to neural networks which are computationally intensive and approximate at best. For real-world autonomous systems, we cannot afford _any_ under-approximation of our system dynamics; otherwise, we'll enjoy the complimentary ride that travels on the superhighway of disgrace to hell.

+ Why Consider this CLF correction scheme for stabilizing trajectories over `statistical learning` methods or `dynamic movement primitives`?

  -    Dynamic Movement Primitives are do not do well when it comes to learning multiple demos;

  -   Statistical Learning approaches, on the other hand, really do not have a guaranteed way of ensuring the learned dynamics are Lyapunov stable;

+ What is different between this and the matlab implementation?

  -  Well, for starters, a cleaner implementation of the Gaussian mixture models/regressors used in estimating dynamics along every trajectory sample.

  - A straightforward CLF learner.

  - A straightforward nonlinear optimization of the CLF cost that handles both _inequality and equality_  constraints.

  - Written in python and easy to port to other open-source robot libraries.

### Methods

+ Through a clever re-parameterization of robot trajectories, by a so-called weighted sum of asymmetric quadratic functions (WSAQF), and nonlinear optimization, we learn stable Lyapunov attractors for the dynamics of a robot's reaching motion, such that we are guaranteed to settle to non-spurious and stable attractors after optimization;

+ This code leverages a control Lyapunov function in deriving the control laws used to stabilize spurious regions of attractors, and eradicate limit cycles that e.g. expectation-maximization may generate;

+ This code is pretty much easy to follow and adapt for any dynamical system. Matter-of-factly, I used it in learning the dynamics of the Torobo 7-DOF arm in Tokyo ca Summer 2018.

### TODOs

+ Add quivers to Lyapunov Function's level sets plot.

+ Add options for plotting different level sets for the Lyapunov Function.

+ Intelligently initialize the Lyapunov Function so optimization iterations do not take such a long time to converge.

+ Fix bug in WSAQF when `L` is started above `1` for a demo.

### Citation

If you use `LyapunovLearner` in your work, please cite it:
```tex
@misc{LyapunovLearner,
  author = {Ogunmolu, Olalekan and Thompson, Rachel Skye and Pérez-Dattari, Rodrigo},
  title = {{Learning Control Lyapunov Functions}},
  year = {2021},
  howpublished = {\url{https://github.com/lakehanne/LyapunovLearner}},
  note = {Accessed October 20, 2021}
}
```
