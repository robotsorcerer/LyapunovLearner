# Torobo_basic
Basic controller for Torobo.
This is for joint and curent control.


+ This experiment was implemented on the Tokyo Robotics 7-DoF arm.

+ First be sure to install all the dependencies in [requirements.txt](/requirements.txt): pip install -r requirements.txt --user.

+ I expect that the experiment should be reproducible on any other 7-DoF arm.

### Generating data

+ First turn off the current on all the servos of your manipulator arm

+ Then manually move the arm to the world coordinates where you want it to go, recording the joint angles and joint angle velocities in the process. See [Torobo/Takahashi/main.py](Torobo/Takahashi/main.py) for an example in the `set_current` function

+ Then run the `ik_sub` executable in the `trac_ik_torobo` package to generate the cartesian coordinates from the joint space coordinates that you recorded. Be sure to call `save_to_file` as a `true` parameter:

```
    roslaunch toroboarm_seven_bringup bringup_real.launch save_to_file:=true
```

- Note that this uses the [KDL](http://www.orocos.org/kdl) and the [dr_kdl](https://github.com/jettan/dr_kdl) packages. Please download and place them in your catkin `src` folder vbefore catkin building

+ Running the `ik_sub` executable should place the new cartesian coordinate file in your `LyapunovLearner/scripts/data/cart_pos.csv` path.


 + Note that depending on the joint limits of your robot arm and the maximum torque each joint's servo is allowed to accept, your robot might react haphazardly while the learning algorithm is running. Please calibrate your robot to the currents before deploying on the real robot.

 + It is expected that your robot's `urdf` file is uploaded to the ros parameter server under the `param` name `/robot_description`. This would be used by `trac_ik` and `dr_kdl` in computing the real-time IK joint positions of the arm.

 ##### Run the move it launcher.

  + In a separate terminal, launch the `torobo bringup moveit` server

     ```
       roslaunch toroboarm_seven_bringup bringup_real.launch
     ```

 ##### Run the IK calculator and service server executable

 + Then launch the recorded joint angles publisher and run the executor as well as follows:

     ```
       roslaunch trac_ik_torobo torobo.launch
     ```

     - it might be a good idea to turn off the data that gets printed out to terminal. Append `disp:=false` to the `roslaunch` command above

 ##### Running the Lyapunov Learner algorithm

 Here, we will run a Gaussian Mixture Regression on samples of data that we gathered from the robot arm. We will then call the `ik_solver` embedded as a service in the Torobo `ik_sub` executable continually in the while loop located in [LyapunovLearner/scripts/ToroboControl/robot_executor/executor.py](LyapunovLearner/scripts/ToroboControl/robot_executor/executor.py).

 + `cd` into the `scripts` folder of `LyapunovLearner` and run `main.py`. Be sure to do this in a `Python2.7` environment where you have exposure to the ros `setup.bash` or `setup.zsh` file.

  Good luck.

 ### Issues

 #### ROS Service IK Issues

   + If you are having issues with the ik solver, try seeing if this returns anything in terminal:
   ```
       rosservice call /torobo/solve_diff_ik "'desired_vel': {'linear': {'x': 0.0, 'y': 0.1, 'z': 0.2}}"
   ```

   + Otherwise, check to be sure the ik service is actually online.
