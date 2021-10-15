### Introduction

This code largely reimplements Learning CLFs using SEDS paper by Khansari-Zadeh. See original code in [matlab here](https://bitbucket.org/khansari/clfdm/src/master/demo_CLFDM_Learning.m).

### Running Lyapunov Learner

 + Please run in a `python 3.6` environment.

 + If you run the [demo.py](/scripts/demo.py) file with the `w` model, you should obtain a chart similar to this:

   ![results_python](/scripts/docs/energy_levels.png)

or with the `s` model, you should obtain a chart similar to this:

   ![results_python](/scripts/docs/energy_levels_sshape.png)


### Citation

If you used `LyapunovLearner` in your work, please cite it:


```tex
@misc{LyapunovLearner,
  author = {Ogunmolu, Olalekan and Thompson, Rachel Skye and Pérez-Dattari, Rodrigo},
  title = {{Learning Control Lyapunov Functions in Python}},
  year = {2020},
  howpublished = {\url{https://github.com/lakehanne/LyapunovLearner}},
  note = {Accessed February 10, 2020}
}
```

### Issues

+ Please open an issue if you are having trouble running this package.

+ Email: lexilighty@gmail.com
