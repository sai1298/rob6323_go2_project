# rob6323_go2_env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
rob6323_go2_env.py

DirectRLEnv implementation for the ROB6323 Go2 locomotion task.

This module contains:
- Rob6323Go2Env: a DirectRLEnv subclass that wires together the robot articulation,
  terrain, and contact sensor.
- Action handling:
  - Action tensor storage and action history (for action-rate penalties).
  - Manual PD control (Kp/Kd + torque limits) applied as joint effort targets.
- Observations:
  - Base velocities, projected gravity, commanded velocities, joint states, actions,
    and gait clock inputs.
- Rewards:
  - Command tracking (linear XY + yaw rate) with exponential shaping.
  - Action-rate penalty using multi-step action history.
  - Raibert heuristic reward term based on foot placement.
  - Stabilization/regularization terms (orientation, vertical velocity, joint velocity,
    angular velocity XY, torque L2).
  - Foot interaction terms (feet clearance + shaped contact force tracking).
- Terminations:
  - Time-out termination and safety termination based on orientation/height.
- Gait target stepping:
  - Computes clock inputs and desired contact states with smoothing.
- Debug visualization:
  - Renders desired vs. current velocity arrows using VisualizationMarkers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor

# Ensure this import matches your directory structure
from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    """ROB6323 Go2 environment using DirectRLEnv API and manual PD control."""

    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs: Any):
        """
        Initialize buffers, logging accumulators, PD parameters, and gait state.

        Args:
            cfg (Rob6323Go2EnvCfg): Environment configuration.
            render_mode (str | None): Render mode passed to the base env.
            **kwargs (Any): Forwarded to the DirectRLEnv constructor.
        """
        super().__init__(cfg, render_mode, **kwargs)

        # Action buffers
        self._actions: torch.Tensor = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._previous_actions: torch.Tensor = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )

        # Commands: [vx, vy, yaw_rate]
        self._commands: torch.Tensor = torch.zeros(self.num_envs, 3, device=self.device)

        # Episode logging accumulators
        self._episode_sums: dict[str, torch.Tensor] = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "rew_action_rate",  
                "raibert_heuristic",  
                "orient",  
                "lin_vel_z", 
                "dof_vel",  
                "ang_vel_xy",  
                "action_l2",  
                "feet_clearance",  
                "tracking_contacts_shaped_force",  
            ]
        }

        # Action history (num_envs, act_dim, history_len=3)
        self.last_actions: torch.Tensor = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # Manual PD controller parameters
        self.Kp: torch.Tensor = torch.full((self.num_envs, 12), self.cfg.Kp, device=self.device)
        self.Kd: torch.Tensor = torch.full((self.num_envs, 12), self.cfg.Kd, device=self.device)
        self.torque_limits: torch.Tensor = cast(torch.Tensor, self.cfg.torque_limits)

        self.desired_joint_pos: torch.Tensor = torch.zeros(self.num_envs, 12, device=self.device)
        self._applied_torques: torch.Tensor = torch.zeros(self.num_envs, 12, device=self.device)

        # Foot/body indexing (cached on first use)
        self._feet_ids: list[int] = []
        self._foot_names: list[str] = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self._base_id: int | None = None

        # Gait state buffers
        self.gait_indices: torch.Tensor = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.clock_inputs: torch.Tensor = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.desired_contact_states: torch.Tensor = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.foot_indices: torch.Tensor = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self) -> None:
        """
        Create robot, sensors, terrain, clone environments, and add lighting.

        Returns:
            None
        """
        self.robot: Articulation = Articulation(self.cfg.robot_cfg)
        self._contact_sensor: ContactSensor = ContactSensor(self.cfg.contact_sensor)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    @property
    def foot_positions_w(self) -> torch.Tensor:
        """
        Return foot positions in world frame as (num_envs, 4, 3).

        Caches foot body indices on first access to avoid repeated name lookups.

        Returns:
            torch.Tensor: Foot positions in world frame (num_envs, 4, 3).
        """
        if not self._feet_ids:
            for name in self._foot_names:
                ids, _ = self.robot.find_bodies(name)
                self._feet_ids.append(int(ids[0]))
        return self.robot.data.body_pos_w[:, self._feet_ids]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Cache current actions and compute desired joint positions (pre-physics).

        Args:
            actions (torch.Tensor): Policy actions for this step.

        Returns:
            None
        """
        self._actions = actions.clone()

        # Target joint positions: default + scaled action offsets
        self.desired_joint_pos = (
            self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos
        )

    def _apply_action(self) -> None:
        """
        Apply manual PD torques (with saturation) as joint effort targets.

        Returns:
            None
        """
        self._applied_torques = torch.clip(
            self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos)
            - self.Kd * self.robot.data.joint_vel,
            -self.torque_limits,
            self.torque_limits,
        )
        self.robot.set_joint_effort_target(self._applied_torques)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """
        Assemble and return the policy observation vector.

        Returns:
            dict[str, torch.Tensor]: A dict containing the policy observation under key "policy".
        """
        self._previous_actions = self._actions.clone()

        obs: torch.Tensor = torch.cat(
            [
                tensor
                for tensor in (
                    self.robot.data.root_lin_vel_b,
                    self.robot.data.root_ang_vel_b,
                    self.robot.data.projected_gravity_b,
                    self._commands,
                    self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                    self.robot.data.joint_vel,
                    self._actions,
                    self.clock_inputs,  # Tutorial Part 4
                )
                if tensor is not None
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        Compute and return the scalar reward for each environment.

        Returns:
            torch.Tensor: Reward vector of shape (num_envs,).
        """
        self._step_contact_targets()

        # Base tracking (exponential shaping)
        lin_vel_error: torch.Tensor = torch.sum(
            torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]),
            dim=1,
        )
        lin_vel_error_mapped: torch.Tensor = torch.exp(-lin_vel_error / 0.25)

        yaw_rate_error: torch.Tensor = torch.square(
            self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2]
        )
        yaw_rate_error_mapped: torch.Tensor = torch.exp(-yaw_rate_error / 0.25)

        # Action-rate penalties (first and second differences)
        rew_action_rate: torch.Tensor = torch.sum(
            torch.square(self._actions - self.last_actions[:, :, 0]), dim=1
        ) * (self.cfg.action_scale**2)

        rew_action_rate += torch.sum(
            torch.square(
                self._actions
                - 2 * self.last_actions[:, :, 0]
                + self.last_actions[:, :, 1]
            ),
            dim=1,
        ) * (self.cfg.action_scale**2)

        # Raibert heuristic term
        rew_raibert: torch.Tensor = self._reward_raibert_heuristic()

        # Refined stabilization / regularization terms
        rew_orient: torch.Tensor = torch.sum(
            torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1
        )
        rew_lin_vel_z: torch.Tensor = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        rew_dof_vel: torch.Tensor = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
        rew_ang_vel_xy: torch.Tensor = torch.sum(
            torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1
        )

        # Torque L2 penalty
        rew_action_l2: torch.Tensor = torch.sum(torch.square(self._applied_torques), dim=1)

        # Advanced foot interaction terms
        rew_feet_clearance: torch.Tensor = self._reward_feet_clearance()
        rew_tracking_contacts_shaped_force: torch.Tensor = self._reward_tracking_contacts_shaped_force()

        rewards: dict[str, torch.Tensor] = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale,
            "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale,
            "raibert_heuristic": rew_raibert * self.cfg.raibert_heuristic_reward_scale,
            "orient": rew_orient * self.cfg.orient_reward_scale,
            "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale,
            "dof_vel": rew_dof_vel * self.cfg.dof_vel_reward_scale,
            "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale,
            "action_l2": rew_action_l2 * self.cfg.action_l2_reward_scale,
            "feet_clearance": rew_feet_clearance * self.cfg.feet_clearance_reward_scale,
            "tracking_contacts_shaped_force": rew_tracking_contacts_shaped_force
            * self.cfg.tracking_contacts_shaped_force_reward_scale,
        }

        # Update action history (most recent at index 0)
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions

        reward: torch.Tensor = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging: accumulate component values
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute and return termination flags.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - died: Environments that terminated due to failure conditions.
                - time_out: Environments that terminated due to time limit.
        """
        time_out: torch.Tensor = self.episode_length_buf >= self.max_episode_length - 1

        # With feet-only contact sensing, rely on height/orientation checks.
        cstr_upsidedown: torch.Tensor = self.robot.data.projected_gravity_b[:, 2] > 0
        base_height: torch.Tensor = self.robot.data.root_pos_w[:, 2]
        cstr_base_height_min: torch.Tensor = base_height < self.cfg.base_height_min

        died: torch.Tensor = cstr_upsidedown | cstr_base_height_min
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Reset the specified environments and populate episode logs.

        Args:
            env_ids (Sequence[int] | None): Environment indices to reset. If None, resets all.

        Returns:
            None
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = cast(Sequence[int], self.robot._ALL_INDICES)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        self.last_actions[env_ids] = 0.0
        self.gait_indices[env_ids] = 0.0

        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)

        joint_pos: torch.Tensor = self.robot.data.default_joint_pos[env_ids]
        joint_vel: torch.Tensor = self.robot.data.default_joint_vel[env_ids]
        default_root_state: torch.Tensor = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        extras: dict[str, float] = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg: torch.Tensor = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = float(episodic_sum_avg / self.max_episode_length_s)
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = {}
        self.extras["log"].update(extras)

        extras2: dict[str, float] = {}
        extras2["Episode_Termination/base_contact"] = float(
            torch.count_nonzero(self.reset_terminated[env_ids]).item()
        )
        extras2["Episode_Termination/time_out"] = float(
            torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        )
        self.extras["log"].update(extras2)

    def _step_contact_targets(self) -> None:
        """
        Advance gait phase and compute clock inputs and desired contact states.

        Returns:
            None
        """
        frequencies: float = 3.0
        phases: float = 0.5
        durations: torch.Tensor = 0.5 * torch.ones((self.num_envs,), device=self.device)

        self.gait_indices = torch.remainder(
            self.gait_indices + self.step_dt * frequencies, 1.0
        )

        foot_indices_list: list[torch.Tensor] = [
            self.gait_indices + phases,
            self.gait_indices,
            self.gait_indices,
            self.gait_indices + phases,
        ]

        self.foot_indices = torch.remainder(torch.stack(foot_indices_list, dim=1), 1.0)

        warped_indices: torch.Tensor = self.foot_indices.clone()
        for i in range(4):
            fi: torch.Tensor = warped_indices[:, i]
            stance: torch.Tensor = fi < durations
            swing: torch.Tensor = ~stance
            fi[stance] = (fi[stance] / durations[stance]) * 0.5
            fi[swing] = (
                0.5 + ((fi[swing] - durations[swing]) / (1 - durations[swing])) * 0.5
            )
            warped_indices[:, i] = fi

        self.clock_inputs = torch.sin(2 * np.pi * warped_indices)

        kappa: float = 0.07
        cdf = torch.distributions.normal.Normal(0, kappa).cdf

        def smooth(fi: torch.Tensor) -> torch.Tensor:
            """
            Smooth a phase signal into a soft stance/swing indicator.

            Args:
                fi (torch.Tensor): Phase values.

            Returns:
                torch.Tensor: Smoothed contact probability-like signal.
            """
            return cdf(torch.remainder(fi, 1.0)) * (
                1 - cdf(torch.remainder(fi, 1.0) - 0.5)
            ) + cdf(torch.remainder(fi, 1.0) - 1.0) * (
                1 - cdf(torch.remainder(fi, 1.0) - 1.5)
            )

        for i in range(4):
            self.desired_contact_states[:, i] = smooth(warped_indices[:, i])

    def _reward_raibert_heuristic(self) -> torch.Tensor:
        """
        Compute Raibert-style foot placement error in the body-yaw frame.

        Returns:
            torch.Tensor: Per-env Raibert error (num_envs,).
        """
        foot_rel: torch.Tensor = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        foot_body: torch.Tensor = torch.zeros(self.num_envs, 4, 3, device=self.device)

        for i in range(4):
            foot_body[:, i, :] = math_utils.quat_apply_yaw(
                math_utils.quat_conjugate(self.robot.data.root_quat_w),
                foot_rel[:, i, :],
            )

        desired_width: float = 0.25
        desired_length: float = 0.45

        ys_nom: torch.Tensor = torch.tensor(
            [
                desired_width / 2,
                -desired_width / 2,
                desired_width / 2,
                -desired_width / 2,
            ],
            device=self.device,
        ).unsqueeze(0)

        xs_nom: torch.Tensor = torch.tensor(
            [
                desired_length / 2,
                desired_length / 2,
                -desired_length / 2,
                -desired_length / 2,
            ],
            device=self.device,
        ).unsqueeze(0)

        phases: torch.Tensor = torch.abs(1.0 - self.foot_indices * 2.0) - 0.5
        freq: float = 3.0

        x_vel: torch.Tensor = self._commands[:, 0:1]
        yaw_vel: torch.Tensor = self._commands[:, 2:3]
        y_vel: torch.Tensor = yaw_vel * desired_length / 2

        ys_offset: torch.Tensor = phases * y_vel * (0.5 / freq)
        ys_offset[:, 2:4] *= -1
        xs_offset: torch.Tensor = phases * x_vel * (0.5 / freq)

        ys_des: torch.Tensor = ys_nom + ys_offset
        xs_des: torch.Tensor = xs_nom + xs_offset

        desired: torch.Tensor = torch.cat((xs_des.unsqueeze(2), ys_des.unsqueeze(2)), dim=2)
        err: torch.Tensor = torch.abs(desired - foot_body[:, :, 0:2])
        return torch.sum(torch.square(err), dim=(1, 2))

    def _reward_feet_clearance(self) -> torch.Tensor:
        """
        Penalize feet that are too low during swing.

        Returns:
            torch.Tensor: Per-env feet clearance penalty (num_envs,).
        """
        target_height: float = 0.1
        is_swing: torch.Tensor = self.desired_contact_states < 0.5
        foot_z: torch.Tensor = self.foot_positions_w[:, :, 2]
        delta: torch.Tensor = target_height - foot_z
        penalty: torch.Tensor = torch.square(torch.clip(delta, min=0.0)) * is_swing
        return torch.sum(penalty, dim=1)

    def _reward_tracking_contacts_shaped_force(self) -> torch.Tensor:
        """
        Reward matching desired contact forces based on desired contact schedule.

        Returns:
            torch.Tensor: Per-env shaped contact-force tracking reward (num_envs,).
        """
        foot_forces: torch.Tensor = self._contact_sensor.data.net_forces_w
        foot_forces_norm: torch.Tensor = torch.norm(foot_forces, dim=-1)

        robot_weight_approx: float = 12.0 * 9.81
        nominal_force: float = robot_weight_approx / 2.0

        desired_forces: torch.Tensor = self.desired_contact_states * nominal_force
        error: torch.Tensor = foot_forces_norm - desired_forces
        return torch.exp(-torch.sum(torch.square(error), dim=1) / nominal_force)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """
        Toggle debug markers for commanded/current velocity visualization.

        Args:
            debug_vis (bool): Whether to enable debug visualization.

        Returns:
            None
        """
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(
                    self.cfg.goal_vel_visualizer_cfg
                )
                self.current_vel_visualizer = VisualizationMarkers(
                    self.cfg.current_vel_visualizer_cfg
                )
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event: Any) -> None:
        """
        Render desired and current XY velocity arrows at the robot base.

        Args:
            event (Any): Visualization callback event payload (unused).

        Returns:
            None
        """
        if not self.robot.is_initialized:
            return

        base_pos_w: torch.Tensor = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self._commands[:, :2]
        )
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b[:, :2]
        )

        self.goal_vel_visualizer.visualize(
            base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale
        )
        self.current_vel_visualizer.visualize(
            base_pos_w, vel_arrow_quat, vel_arrow_scale
        )

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert XY velocity vectors into marker scale and orientation quaternions.

        Args:
            xy_velocity (torch.Tensor): XY velocity vectors (num_envs, 2).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - arrow_scale: Marker scale tensors (num_envs, 3).
                - arrow_quat: Marker orientation quaternions (num_envs, 4).
        """
        default_scale = cast(Any, self.goal_vel_visualizer.cfg.markers["arrow"].scale)
        arrow_scale: torch.Tensor = torch.tensor(default_scale, device=self.device).repeat(
            xy_velocity.shape[0], 1
        )

        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        heading_angle: torch.Tensor = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros: torch.Tensor = torch.zeros_like(heading_angle)

        arrow_quat: torch.Tensor = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w: torch.Tensor = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
