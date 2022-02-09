SEDS Package: version 1.95 issued on 12 February 2013

This packages contains the CLFDM algorithm presented in the following paper:

S.M. Khansari-Zadeh and A. Billard (2014), "Learning Control Lyapunov Function
to Ensure Stability of Dynamical System-based Robot Reaching Motions." 
Robotics and Autonomous Systems, vol. 62, num 6, p. 752-765.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%         Copyright (c) 2014 Mohammad Khansari, LASA Lab, EPFL,       %%%
%%%          CH-1015 Lausanne, Switzerland, http://lasa.epfl.ch         %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The program is free for non-commercial academic use. Please contact the
author if you are interested in using the software for commercial purposes.
The software must not be modified or distributed without prior permission
of the authors. Please acknowledge the authors in any academic publication
that have made use of this code or part of it. Please use the aforementioned
paper for reference.

To get latest update of the software please visit
                          http://cs.stanford.edu/people/khansari/

Please send your feedbacks or questions to:
                          khansari_at_cs.stanford.edu


This source code include two matlab functions: 'demo_Plot_Results.m' and 
'demo_CLFDM_Learning.m', and 4 subdirectories: 'CLFDM_lib', 'regress_gauss_mix_lib', 
'ExampleModels', and 'Doc'.

demo_CLFDM_Learning: a matlab script illustrating how to use CLFDM_lib to learn
                     an arbitrary model from a set of demonstrations.

CLFDM_lib: contains code which implements CLFDM. See the document 'Doc/SEDS_Slides.pdf' 
           for further details about this library.

regress_gauss_mix_lib: A library for Gaussian Mixture Model. We use this library just for illustrative
         purpose. Feel free to use any library that you want for encoding the motion.

ExampleModels: contains two handwriting motions recorded from Tablet-PC.

Doc: It includes Khansari_Billard_RAS2014.pdf, which is the original paper on CLFDM.

When running the demos, it is assumed that your current directory is the
CLFDM_package directory. Otherwise, you should manually add both the 'CLFDM_lib'
and 'regress_gauss_mix_lib' directories to the matab path.