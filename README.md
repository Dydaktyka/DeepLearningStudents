# DeepLearning
Materials for my Deep Learning course 


## Setting up the python environment

As in most of the machine learning, you will be working with python using jupyter notebooks or jupyter lab. So first you have to set up a proper python environment. We strongly encourage you to use some form of a virtual environment. We recommend the Anaconda or its smaller subset miniconda.

After installing anaconda or miniconda create a new virtual environment deeplearning (or any other name you want):

```
conda create -n deeplearning python=3.7
```
Then activate the environment
```
conda activate deeplearning
```
Now you can install required packages (if you are using Anaconda some maybe already installed):

```
conda install  jupyter jupyterlab
conda install numpy scipy scikit-learn matplotlib
```
You will also need PyTorch. The installation instruction are here. If you have a relatively new NVidia GPU you should install a GPU version.

If you want to keep your notebooks under version control which we strongly suggest doing, you may want to use jupytext. To use jupytext install packages jupytext and nodejs with

```
conda install jupytext nodejs
```
and then run
```
jupyter lab build
```
How to use jupytext with with git version control is described here.
