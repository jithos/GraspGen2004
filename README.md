# GraspGen2004 — ROS1 wrapper for GraspGen project

This repository contains the implementation of a ROS1 wrapper around the diffusion-based grasping framwork GraspGen <https://github.com/NVlabs/GraspGen>.
The system runs inside a Docker container and provides GPU-accelerated grasp synthesis together with visualization via MeshCat.

## 1. OVERVIEW

### The project consists of:

- GraspGen wrapper
- ROS Noetic integration
- Robokudo message interfaces
- MeshCat visualization
- Docker-based reproducible environment


## 2. REQUIREMENTS

- NVIDIA GPU
- NVIDIA drivers installed
- NVIDIA Container Toolkit
- Docker


## 3. INSTALLATION

### 1. Clone Repository

IMPORTANT: This project uses submodules.

```shell
git clone --recurse-submodules <repo-url>
cd GraspGen2004
```
If already cloned without submodules:

```shell
git submodule update --init --recursive
```

### 2. Build Docker Image

Build the full environment:

```shell
./docker/build.sh
```

This builds:

- PyTorch + CUDA environment
- GraspGen dependencies
- ROS Noetic (ros-base)
- Python dependencies

Build time may take 20–40 minutes.


## 4. BUILDING ROS WORKSPACE

After starting the container (see "Running the System - Terminal 1 — Start Docker Container" below):

```shell
cd catkin_ws
catkin build
```
```shell
source devel/setup.bash
```

This step only needs to be done once unless ROS packages change. Sourcing is not required again if the container ist re-started.

## 5. RUNNING THE SYSTEM

The system requires two terminals. Assuming you are now in repository root.


### Terminal 1 — Start Docker Container

If the container is already running:

```shell
./docker/start.sh
```

Else run the following:

```shell
./docker/run.sh
```

This will:

- Start container with GPU access
- Mount project directory
- Configure ROS networking
- Install python package in editable mode

You are now inside the container shell.


### Terminal 2 — Start MeshCat Server

Open a new terminal and run:

```shell
./docker/run_meshcat.sh
```

This starts the visualization server.


### RUNNING THE ROS NODE

The roscore is assumed to be already running and is managed by the robot/external outer node. The Terminal 2 and the meshcat-server must be run first.
Inside the container (Terminal 1):

```shell
rosrun graspgen_grasping graspgen_ros_wrapper.py
```

### MESHCAT VISUALIZATION

MeshCat runs automatically once the server is started. A link to show the visualization in the browser should be visible in Terminal 2.


## 6. CONFIGURATIONS TO MAKE IN GRASPING PIPELINE

### Configuration adaption
In the Grasping Pipelines config.yaml the _grasppoint_estimator_topic_ needs to be set to  `/pose_estimator/find_grasppose_graspgen`.

### Direct pose estimation
In grasp_method_selector.py, there might be adaptions neccassary to get `'direct_grasp'` returned as only then the GraspGen-wrapper is called.


# GraspGenforHSR

```shell
rosrun graspgen_grasping hsrb_grasping.py
```

### Configuration adaption
In the Grasping Pipelines config.yaml the _grasppoint_estimator_topic_ needs to be set to  `/pose_estimator/find_grasppose_hsrb_graspgen`.



