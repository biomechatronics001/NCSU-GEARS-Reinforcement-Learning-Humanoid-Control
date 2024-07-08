# Program Overview 

**Duration**: 07/08-07/30

**Instructor**: Professor Hao Su, Ivan Sanchez (postdoc), Junxi Zhu (PhD student)

**Location**: Room 3209, Engineering Building III (unless otherwise specified; for in-person students only)

**Time**: TBD

**Zoom link**: https://ncsu.zoom.us/my/junxizhu

**Topics**:

* Reinforcement learning basics (including a cart-pole code example on Google Colab)
* Humanoid robot simulation project using reinforcement learning

# GEARS Program Schedule

![GEARS Schedule 1](./resources/GEARS_Schedule_1.png) 

![GEARS Schedule 2](./resources/GEARS_Schedule_2.png)

# Tentative Syllabus for Individual Project & Training

| Week | Lecture | Content | Homework Assignment | 
| :---: | :---: | --- | --- |
| 1 | 1 | Program overview (07/11, 10a - 11:30a, New York time, Room 3209, Enginnering Building III) | See [Paper List](#paper-list) |
| 1 | 2 | Install software environment (Ubuntu, MuJoCo, IsaacGym) (07/12, 10a - 11:30a, New York time, Room 3209, Enginnering Building III) |  |
| 2 | 3 | Reinforcement learning basics<br>Cart-pole example (Date TBD) | See [Homework 1 Cart-Pole Example](#homework-1-cart-pole-example) | 
| 2 | 4 | Define custom URDF file for robots and visualize in MuJuCo (Date TBD) |  |
| 3 | 5 | Introduction to humanoid robots<br>Reward function formulation (Date TBD) | See [Homework 2 Reward Function Formulation](#homework-2-reward-function-formulation)
| 3 | 6 | Tune reward function and train humanoid robot controller (Date TBD) | See [Homework 3 Humanoid Virtual Competition](#homework-3-humanoid-virtual-competition) |
| 4 | 7 | Poster feedback | Poster must be completed by July 25 |


# Hardware and Software Requirement

It is strongly recommended that you have a computer with **Linux 18.04** operating system and a **Nvidia GPU** that supports CUDA with **more than 8 GB of VRAM**. They are necessary for the reinforcement learning project.

For in-person students, if you are not able to meet this requirement, you may use the computer we provide in the lab. Two students form a group and share one computer. A total of four computers are available in the lab.

For remote students, you may use the Virtual Computing Lab (VCL) facilities provided by the university. Details TBD.

# Paper List

* **[Humanoid]** S. H. Jeon, S. Heim, C. Khazoom, and S. Kim, “Benchmarking Potential Based Rewards for Learning Humanoid Locomotion,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), London, United Kingdom, May 2023, pp. 9204–9210.
* **[Hutter19 Science Robotics]** J. Hwangbo, J. Lee, A. Dosovitskiy, D. Bellicoso, V. Tsounis, V. Koltun, and M. Hutter, “Learning agile and dynamic motor skills for legged robots,” Science Robotics, vol. 4, no. 26, p. eaau5872, Jan. 2019.
* **[Hutter20 Science Robotics]** J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, “Learning quadrupedal locomotion over challenging terrain,” Science Robotics, vol. 5, no. 47, p. eabc5986, Oct. 2020.
* **[ETH Hutter 2022 ICRA]** [Legged Robots on the way from subterranean](https://www.youtube.com/watch?v=XwheB2_dyMQ)
* **[Our 2024 Nature]** S. Luo, M. Jiang, S. Zhang, J. Zhu, S. Yu, I. Dominguez Silva, T. Wang, E. Rouse, B. Zhou, H. Yuk, X. Zhou, and H. Su, “Experiment-free exoskeleton assistance via learning in simulation,” Nature, vol. 630, no. 8016, pp. 353–359, Jun. 2024.
* **[Davide23 Nature]** E. Kaufmann, L. Bauersfeld, A. Loquercio, M. Müller, V. Koltun, and D. Scaramuzza, “Champion-level drone racing using deep reinforcement learning,” Nature, vol. 620, no. 7976, pp. 982–987, Aug. 2023. | [Seminar Talk](https://www.youtube.com/watch?v=tb1SCib0OTo)
* **[IIya Deep Learning]** [Deep Learning Theory Session Ilya SutskeverIlya Sutskever](https://www.youtube.com/watch?v=OAm6zyR_c8k)

<br>

# Cart-Pole Example

## Objective

1. Understand the basic concepts in reinforcement learning:
  * Environment
  * Agent
  * Satates
  * Observation
  * Action, reward, policy, etc.

2. Investigate the influence of number of neurons, number of network hidden layers, and training episodes on the performance of the trained policy.

## Introduction

For this activity we will employ Google Colab togheter with OpenAI's Gym and Python.

### Google Colab

Google Colab, or Google Colaboratory, is a free cloud service provided by Google that allows users to write and execute Python code in a web-based interactive environment.
It offers convenient features such as:

* Free Access to GPUs and TPUs: free access to powerful GPUs and TPUs, making it easier to train machine learning models and perform computations that require significant processing power.

* Interactive Python Notebooks: Colab uses Jupyter Notebook, which supports rich text, visualizations, and code execution in an interactive format.

* Easy Sharing and Collaboration: Notebooks can be easily shared and collaborated on, similar to Google Docs. Multiple users can work on the same notebook simultaneously.

* Integration with Google Drive: Colab integrates seamlessly with Google Drive, allowing you to save and access your notebooks and datasets directly from your Drive.

* Code Execution in the Cloud: Since the code runs on Google's cloud servers, users don't need to worry about the computational limits of their local machines.

* Markdown Support: Users can include formatted text, equations, and visualizations within the notebook using Markdown, enhancing the readability and presentation of the code and results.

### OpenAI Gym

OpenAI Gym is a toolkit developed by OpenAI for developing and comparing reinforcement learning (RL) algorithms. It provides a standardized set of environments (e.g., games, robotic tasks, control problems) that researchers and developers can use to test and evaluate their RL algorithms. Here are some key features of OpenAI Gym:

* Variety of Environments: Gym offers a wide range of environments, including classic control tasks, Atari games, robotic simulations, and more.

* Standard Interface: All environments in Gym follow a standard interface with methods like reset(), step(action), render(), and close(). This makes it easy to switch between different environments without changing the algorithm code.

* Community and Benchmarks: Gym has a strong community of researchers and practitioners who contribute to the toolkit, create new environments, and share results. It also provides benchmarks for comparing the performance of different RL algorithms.

* Integration with Other Libraries: Gym can be easily integrated with popular RL libraries like TensorFlow, PyTorch, and stable-baselines, facilitating the development and testing of RL models.

* Extensibility: Users can create custom environments by following the Gym API, making it suitable for a wide range of applications beyond the provided environments.

### CartPole Environment

OpenAI's Gym environment "CartPole" is a classic control problem in reinforcement learning (RL). The primary objective is to balance a pole on a cart by applying forces to the cart to keep the pole upright. Here's a detailed description of the CartPole environment:

Environment Setup

1. Cart and Pole Dynamics:

  * Cart: A small cart that can move left or right along a frictionless track.
  * Pole: A pole attached to the cart by an unactuated joint. The pole starts upright and can fall to either side.

2. State Space:

  * The state is represented by a four-dimensional vector:
    1. cart_position: The position of the cart on the track.
    2. cart_velocity: The velocity of the cart.
    3. pole_angle: The angle of the pole from the vertical.
    4. pole_velocity_at_tip: The velocity of the tip of the pole.

3. Action Space:

* The action space is discrete with two possible actions:
    * 0: Apply a force to the cart to move it to the left.
    * 1: Apply a force to the cart to move it to the right.

Objective
The primary goal is to keep the pole balanced and the cart within the track boundaries for as long as possible. The episode terminates if:

The pole angle exceeds ±12 degrees from the vertical.
The cart position exceeds ±2.4 units from the center.
The episode length reaches a maximum of 200 time steps (configurable in some versions).
Reward
The reward structure is simple:

The agent receives a reward of +1 for every time step the pole remains upright and within the allowed boundaries.
Termination Conditions
The episode ends when:

The pole falls beyond the allowed angle.
The cart moves out of the allowed position range.
The maximum number of time steps is reached.

## Set up

On this regard, the first step is to create a new Google Colab Notebook

# Homework 1 Cart-Pole Example

Placeholder

# Humanoid Project

## Objective

1. Have a deeper understanding of the concepts in reinforcement learning through this more advanced example.
2. Create a URDF file for a custom robot.
3. Tune the reward function and understand the influence of different reward terms on the controller performance. 

**Paper**: S. H. Jeon, S. Heim, C. Khazoom, and S. Kim, “Benchmarking Potential Based Rewards for Learning Humanoid Locomotion,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), London, United Kingdom, May 2023, pp. 9204–9210.


## Installation

* Create the virtual environment using Python 3.8 (``user`` is your machine username; you may also use Anaconda to create and manage the virtual environment)
  
  ```bash
  virtualenv /home/user/leggedrobot --python=python3
  ```

* Activate the virtual environment

  ```bash
  source /home/user/leggedrobot/bin/activate
  ```

   **Note**: You should see the name of the active virtual environment in parenthesis at the begining of the line.
      Something like this ``(leggedrobot)user@PCname:~$``.
    

* Install required libraries (pythorch 1.10 and cuda 11.3)

  ```bash
  pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
  ```

* Install Isaac Gym

  1. Download Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
  2. Extract the zip package in the virtual environment folder.
  3. ``cd isaacgym_lib/python && pip install -e .`` to install the requirements.
  4. Test the installation by running an example: ``cd isaacgym/python/examples && python 1080_balls_of_solitude.py``.
  
  **Note**:: You should be able to see a new window apperaing with a group of balls falling
  
  ![IsaacGymDemo](./resources/balls_of_solitude.png)
  
* Clone the pbrs-humanoid repository and initialize the submodules
  
  1. ``git clone https://github.com/se-hwan/pbrs-humanoid.git``
  2. ``cd pbrs-humanoid/gpugym && git submodule init && git submodule update``

  **Note**: In case you dont have git installed: ``sudo apt-get install git``. Then, clone the repository.
  
* Install gpu_rl (Proximal Policy Optimization - PPO implementation)

  ``cd pbrs-humanoid/gpu_rl && pip install -e .``

* Install gpuGym

  ``cd .. && pip install -e .``

* Install WandB (for tracking on the learned policy during the training stage)

  ``pip install wandb==0.15.11``

## Setup Wandb for logging

1. Create an account on https://wandb.ai/site.

2. Copy your personal API key from the homepage.

    ![WandbHomepage](./resources/Wandb_Homepage.png)

3. Create a new project from the homepage.

4. In the virtual environment, execute the following and enter your API key and prese Enter. Note that for security reasons, the key you entered/pasted will not be visible.

    ```python
    wandb login
    ```

## Train the control policy

To start the training, exeute the following:

```python
python gpugym/scripts/train.py --task=pbrs:humanoid --experiment_name=<NAME> --run_name=<NAME> --wandb_project=<NAME> --wandb_entity=<NAME> --wandb_name=<NAME>
```
  
**Note**: You should see something like this
![HumanoidTraining](./resources/PBRS_MIT_Humanoid_Training.png)
* To run on CPU add following arguments: --sim_device=cpu, --rl_device=cpu (sim on CPU and rl on GPU is possible).
* To run headless (no rendering) add --headless.
* Important: To improve performance, once the training starts press v to stop the rendering. You can then enable itlaterto check the progress.
* The trained policy is saved in gpugym/logs/[experiment_name]\/[date_time]\_[run_name]\/model_[iteration].pt, where [experiment_name] and [run_name] are defined in the train config.
* The following command line arguments override the values set in the config files:
  * ``--task`` TASK: Task name.
  * ``--resume`` Resume training from a checkpoint
  * ``--experiment_name`` EXPERIMENT_NAME: Name of the experiment to run or load.
  * ``--run_name`` RUN_NAME: Name of the run.
  * ``--load_run`` LOAD_RUN: Name of the run to load when resume=True. If -1: will load the last run.
  * ``--checkpoint`` CHECKPOINT: Saved model checkpoint number. If -1: will load the last checkpoint.
  * ``--num_envs`` NUM_ENVS: Number of environments to create.
  * ``--seed`` SEED: Random seed.
  * ``--max_iterations`` MAX_ITERATIONS: Maximum number of training iterations.
  * ``-wandb_project`` WANDB_PROJECT: Project name on Wandb.
  * ``wandb_entity`` WANDB_ENTITY: This is your Wandb user name from the homepage.
  * ``wandb_name`` WANDB_NAME: This is the display name of your run on Wandb.

## Run the trained policy

  ``python gpugym/scripts/play.py --task=pbrs:humanoid``

  **Note**: This is the result: https://www.youtube.com/watch?v=4AzTJMkW2ZA

  * By default the loaded policy is the last model of the last run of the experiment folder.
  * Other runs/model iteration can be selected by setting ``load_run`` and ``checkpoint`` in the train config.

# Homework 2 Reward Function Formulation

Placeholder

# Homework 3 Humanoid Virtual Competition

Placeholder


