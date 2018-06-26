### Introduction

+ This experiment was implemented on the Tokyo Robotics 7-DoF arm.

+ I expect that the experiment should be reproducible on any other 7-DoF arm.

### Generating data_type

+ First turn off the current on all the servos of your manipulator arm

+ Then manually move the arm to the world coordinates where you want it to go, recording the joint angles and joint angle velocities in the process. See [Torobo/Takahashi/main.py](Torobo/Takahashi/main.py) for an example in the `set_current` function

+ Then run the `ik_sub` executable in the `trac_ik_torobo` package to generate the cartesian coordinates from the joint space coordinates that you recorded
    - Note that this uses the [KDL](http://www.orocos.org/kdl) and the [dr_kdl](https://github.com/jettan/dr_kdl) library packages. Please download and place them in your catkin `src` folder


+ Running the ik_sub should place the new cartesian coordinate file in your `LyapunovLearner/scripts/data/cart_pos.csv` path.

+ Voila! You are ready to run Lyapunov learner.

### Runing Lyapunov Learner

+ `cd` into the `scripts` folder of `LyapunovLearner` and run `main.py`. This should take care of business.

### Issues
+ Please open an issue in this repo if you are having trouble running this package.

+ Email: lexilighty@gmail.com
