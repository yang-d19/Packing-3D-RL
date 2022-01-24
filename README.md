# Packing-3D-RL

### 1. Introduction to the project

This is a project for solving bin packing problems, which use methods like heuristic searching and reinforcement learning. 

The demo using heuristic searching method has been completed, you can run `test.py` to see how it works. The part of reinforcement learning hasn't been completed and I can't guarantee when can I finish it. (Maybe  never)

### 2. Functions of each file

Below are explanations to some files:

1. `utils.py` provides basic classes that many other files may use, including `Position`, `Attitude`, `Transform` and other classes concerning the status of a single object.

2. `object.py` provides definition of class `Geometry`  and `Item`, `Geometry` mainly contains geometric transformation and heightmap calculation. `Item` wraps up `Geometry` , makes it more accessible to upper classes.

3. `container.py` defines class `Container`, it provides manipulations of searching possible positions and adding an item to the container.

4. `pack.py` is the top-level class, it defines how to solve a bin packing problem, it provides functions about loading a sequence of objects, automatically adding them to a container, and clearing the container.

5. `show.py` defines class `Display`, it visualizes the procedure of bin packing decision. You only need to pass an object of type `Geometry` to its function `show3d()`, then it will show you how this geometry looks like.

6. `test.py` provides a demo that use heuristic method to solve bin packing problem. If it runs successfully,  you will see the window as below. 

   <img src="pictures\demo.png" alt="demo" style="zoom:20%;" />

7. `train.py`, `env.py` and `common/`, `DQN/` are used for the reinforcement learning method, but it hasn't complete, so these files and folders remain unused.

### 3. Anaconda environment

I have exported my anaconda environment to the file `environment.yaml`, use `conda env create -f environment.yaml` to copy my environment to your computer.

### 4. The idea

This idea of this project comes from the paper ***Stable bin packing of non-convex 3D objects with a robot manipulator, Fan Wang and Kris Hauser.***  *(arXiv:1812.04093v1 [cs.RO] 10 Dec 2018)* , and this paper is attached in the `literature` folder, you can check it for detailed information.

### 5. About author

Yang Ding

Department of Automation, Tsinghua University

Email: yangding19thu@163.com

Web: https://yang-d19.github.io/
