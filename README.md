# Kernel exponential families

[![Build Status](https://travis-ci.org/karlnapf/kernel_exp_family.png)](https://travis-ci.org/karlnapf/kernel_exp_family)
[![Coverage Status](https://coveralls.io/repos/karlnapf/kernel_exp_family/badge.svg?branch=master&service=github)](https://coveralls.io/github/karlnapf/kernel_exp_family?branch=master)

Various estimators of the [infinite dimensional exponential family model](http://arxiv.org/abs/1312.3516). In particular, effecient approximations from our NIPS 2015 paper on [Gradient-free Hamiltonain Monte Carlo with Effecient Kernel Exponential Families](http://arxiv.org/abs/1506.02564).

For learning parameters, there is the option to use the Bayesian optimisation package [pybo](https://github.com/mwhoffman/pybo).

Install dependencies:

    pip install -r https://raw.githubusercontent.com/karlnapf/kernel_exp_family/master/requirements.txt
    
Install ```kernel_exp_family```:

    pip install git+https://github.com/karlnapf/kernel_exp_family.git

A list of examples can be found [here](kernel_exp_family/examples). For example, run

    python -m kernel_exp_family.examples.demo_simple.py

