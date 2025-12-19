# ROB6323 Go2 Locomotion — DirectRLEnv (Isaac Lab)

This project implements a custom DirectRLEnv for quadruped locomotion using the
Unitree Go2 robot in Isaac Lab. The environment is designed for stable learning,
clear reward structure, and reproducible experimentation.

The implementation includes:
- Manual PD torque control with implicit actuator PD disabled
- Feet-only contact sensing for consistent force indexing
- Gait-phase clock inputs and desired contact state shaping
- Velocity tracking rewards with stability and smoothness penalties
- Raibert heuristic foot placement term
- Explicit episodic reward logging and clean termination handling

This README documents all major changes, the rationale behind them, and exact
reproduction steps with fixed seeds, as required for grading.

---

## 1. Major Changes Summary

### 1.1 Manual PD Torque Control

Actions are interpreted as joint position offsets around the default joint pose.
Joint torques are computed manually using a PD controller:

τ = Kp (q_des − q) − Kd q̇

Torques are clipped to joint torque limits before being applied.

Location:
- `rob6323_go2_env.py`
  - `_pre_physics_step()` constructs desired joint positions
  - `_apply_action()` computes and applies PD torques

Rationale:
Manual PD control provides stable low-level actuation and avoids interference
from simulator-side controllers.

---

### 1.2 Disabling Implicit Actuator PD

Implicit actuator stiffness and damping are set to zero to avoid double PD
control when manual PD torques are applied in the environment.

Location:
- `rob6323_go2_env_cfg.py`
  - `ImplicitActuatorCfg(stiffness=0.0, damping=0.0)`

Rationale:
Prevents controllers from fighting each other and significantly improves motion
smoothness and learning stability.

---

### 1.3 Feet-only Contact Sensor Configuration

The contact sensor is restricted to foot links only so that sensor body indices
map consistently to the four feet.

Mapping:
- Index 0: Front-left foot
- Index 1: Front-right foot
- Index 2: Rear-left foot
- Index 3: Rear-right foot

Location:
- `rob6323_go2_env_cfg.py`
  - `ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*_foot")`

Rationale:
Avoids index mismatch bugs caused by including non-foot bodies and enables
reliable contact-force-based rewards.

---

### 1.4 Gait Phase Clocks and Desired Contact States

A gait phase variable is maintained and used to generate:
- Sinusoidal clock inputs for each foot
- Binary desired contact states based on duty factor

Location:
- `rob6323_go2_env.py`
  - `_step_contact_targets()`

Rationale:
Provides rhythmic structure for locomotion and enables contact-aware reward
shaping.

---

### 1.5 Reward Function Design

The reward is composed of multiple terms:
- Linear velocity tracking in the horizontal plane
- Yaw rate tracking
- Action rate and second-difference smoothness penalties
- Orientation and stability penalties
- Joint velocity and angular velocity penalties
- Torque L2 regularization
- Swing-foot clearance penalty
- Contact-force tracking reward
- Raibert heuristic foot placement penalty

Location:
- `rob6323_go2_env.py`
  - `_get_rewards()`
  - `_reward_raibert_heuristic()`
  - `_reward_feet_clearance()`
  - `_reward_tracking_contacts_shaped_force()`

Rationale:
Encourages commanded motion while penalizing unstable, inefficient, or jerky
behavior and improving gait quality.

---

### 1.6 Episodic Logging and Reset Handling

Each reward term is accumulated per episode and logged on reset, normalized by
episode duration. Action history and gait buffers are reset correctly for each
environment.

Location:
- `rob6323_go2_env.py`
  - `_reset_idx()`

Rationale:
Provides transparent diagnostics for training analysis and grading validation.

---

## 2. Key Configuration Parameters

Defined in `rob6323_go2_env_cfg.py`:
- Simulation timestep: `dt = 1/200`
- Control decimation: `decimation = 4`
- Episode length: `20.0 s`
- Action scaling: `0.25`
- PD gains: `Kp = 20.0`, `Kd = 0.5`
- Torque limits: `23.5 Nm`
- Termination base height threshold: `0.20 m`

Reward scales are defined explicitly at the bottom of the configuration file.

---

## 3. Reproduction Instructions

All results reported were obtained using fixed random seeds. The primary seed
used for experiments is 42.

To run training and generate logs, please follow the instructions provided in
the official ROB6323 Go2 project repository:

https://github.com/machines-in-motion/rob6323_go2_project?tab=readme-ov-file

All commands, logging behavior, and seed handling used in this project are
consistent with the procedures described in the above repository.


### 3.1 Environment Setup

The environment setup follows the official ROB6323 Go2 project instructions.

Please refer to the following tutorial for detailed installation and setup steps:
https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md

All experiments and results reported in this project assume the environment has been set up exactly as described in the above tutorial.

