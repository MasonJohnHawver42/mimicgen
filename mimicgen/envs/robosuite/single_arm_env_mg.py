# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import xml.etree.ElementTree as ET
import robosuite
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.utils.mjcf_utils import string_to_array


import robosuite.utils.transform_utils as T
from robosuite.utils.observables import Observable, sensor, create_gaussian_noise_corrupter

import numpy as np

try:
    # only needed for running hammer cleanup and kitchen tasks
    import robosuite_task_zoo
except ImportError:
    pass

import mimicgen


class SingleArmEnv_MG(SingleArmEnv):
            
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        mount_types="default", 
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        force_torque_hist_len=32,
        renderer="mujoco",
        renderer_config=None    
    ):
        
        self.ft_hist_len = force_torque_hist_len

        self.ft_noise_std = np.array([5.0, 0.05], dtype=np.float64)
        self.prop_noise_std = np.array([0.001, 0.01], dtype=np.float64)
        
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types=mount_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )
    """
    Custom version of base class for single arm robosuite tasks for mimicgen.
    """
    def edit_model_xml(self, xml_str):
        """
        This function edits the model xml with custom changes, including resolving relative paths,
        applying changes retroactively to existing demonstration files, and other custom scripts.
        Environment subclasses should modify this function to add environment-specific xml editing features.
        Args:
            xml_str (str): Mujoco sim demonstration XML file as string
        Returns:
            str: Edited xml file as string
        """

        path = os.path.split(robosuite.__file__)[0]
        path_split = path.split("/")

        # replace mesh and texture file paths
        tree = ET.fromstring(xml_str)
        root = tree
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        textures = asset.findall("texture")
        all_elements = meshes + textures

        for elem in all_elements:
            old_path = elem.get("file")
            if old_path is None:
                continue
            old_path_split = old_path.split("/")

            # replace all paths to robosuite assets
            check_lst = [loc for loc, val in enumerate(old_path_split) if val == "robosuite"]
            if len(check_lst) > 0:
                ind = max(check_lst) # last occurrence index
                new_path_split = path_split + old_path_split[ind + 1 :]
                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

            # replace all paths to mimicgen assets
            check_lst = [loc for loc, val in enumerate(old_path_split) if val == "mimicgen"]
            if len(check_lst) > 0:
                ind = max(check_lst) # last occurrence index
                new_path_split = os.path.split(mimicgen.__file__)[0].split("/") + old_path_split[ind + 1 :]
                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

            # note: needed since some datasets may have old paths when repo was named mimicgen_envs
            check_lst = [loc for loc, val in enumerate(old_path_split) if val == "mimicgen_envs"]
            if len(check_lst) > 0:
                ind = max(check_lst) # last occurrence index
                new_path_split = os.path.split(mimicgen.__file__)[0].split("/") + old_path_split[ind + 1 :]
                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

            # replace all paths to robosuite_task_zoo assets
            check_lst = [loc for loc, val in enumerate(old_path_split) if val == "robosuite_task_zoo"]
            if len(check_lst) > 0:
                ind = max(check_lst) # last occurrence index
                new_path_split = os.path.split(robosuite_task_zoo.__file__)[0].split("/") + old_path_split[ind + 1 :]
                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

        return ET.tostring(root, encoding="utf8").decode("utf8")

    def _check_grasp_tolerant(self, gripper, object_geoms):
        """
        Tolerant version of check grasp function - often needed for checking grasp with Shapenet mugs.

        TODO: only tested for panda, update for other robots.
        """
        check_1 = self._check_grasp(gripper=gripper, object_geoms=object_geoms)

        check_2 = self._check_grasp(gripper=["gripper0_finger1_collision", "gripper0_finger2_pad_collision"], object_geoms=object_geoms)

        check_3 = self._check_grasp(gripper=["gripper0_finger2_collision", "gripper0_finger1_pad_collision"], object_geoms=object_geoms)

        return check_1 or check_2 or check_3

    def _add_agentview_full_camera(self, arena):
        """
        Add camera with full perspective of tabletop.
        """
        arena.set_camera(
            camera_name="agentview_full",
            pos=string_to_array("0.753078462147161 2.062036796036723e-08 1.5194726087166726"),
            quat=string_to_array("0.6432409286499023 0.293668270111084 0.2936684489250183 0.6432408690452576"),
        )
    
    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        pf = self.robots[0].robot_model.naming_prefix

        ft_all_denoised_modality = pf + "ft_denoised"
        ft_all_modality = pf + "forcetorque"
        prop_all_modality = pf + "proprioception"

        # print("HEREEEEEE", ft_all_denoised_modality)

        @sensor(modality=ft_all_denoised_modality)
        def ft_all_denoised(obs_cache):
            ft_curr = np.hstack([self.robots[0].ee_force, self.robots[0].ee_torque])
            if "ft_all_denoised" in obs_cache.keys():
                ft_hist = obs_cache['ft_all_denoised']
                
                # Add current ft obs to history
                ft_hist = np.vstack((ft_hist, ft_curr))
                ft_hist = ft_hist[1:]
            else:
                # Pad history to match desired history length
                ft_hist = np.vstack((ft_curr, np.zeros(ft_curr.shape)))
                ft_hist = np.pad(ft_hist, pad_width=((self.ft_hist_len-ft_hist.shape[0]+1,0),), mode='edge')
                ft_hist = ft_hist[:-1, :6]
            # print("WHATTT", ft_hist)
            return ft_hist
        
        # Force-torque data from both arms
        @sensor(modality=ft_all_modality)
        def ft_all(obs_cache):
            ft_curr = np.hstack([self.robots[0].ee_force, self.robots[0].ee_torque])
            if "ft_all" in obs_cache.keys():
                ft_hist = obs_cache['ft_all']
                
                # Add current ft obs to history
                ft_hist = np.vstack((ft_hist, ft_curr))
                ft_hist = ft_hist[1:]
            else:
                # Pad history to match desired history length
                ft_hist = np.vstack((ft_curr, np.zeros(ft_curr.shape)))
                ft_hist = np.pad(ft_hist, pad_width=((self.ft_hist_len-ft_hist.shape[0]+1,0),), mode='edge')
                ft_hist = ft_hist[:-1, :6]
            return ft_hist

        # End-effector site position and rotation from both arms
        @sensor(modality=prop_all_modality)
        def prop_all(obs_cache):
            eef_pos = np.array(self.robots[0].sim.data.site_xpos[self.robots[0].eef_site_id])
            eef_quat = T.convert_quat(self.robots[0].sim.data.get_body_xquat(self.robots[0].robot_model.eef_name), to="xyzw")
            prop_curr = np.hstack([eef_pos, eef_quat])
            return prop_curr

        # Add in gaussian force-torque corrupter
        def ft_corrupter(inp):
            inp_c = np.array(inp)

            # Generate noise 
            force_noise = self.ft_noise_std[0] * np.random.randn(3)
            torque_noise = self.ft_noise_std[1] * np.random.randn(3)

            # Apply noise to measurements
            inp_c[-1,:3] += force_noise[:3]
            inp_c[-1,3:6] += torque_noise[:3]

            return inp_c
        
        # Add in gaussian proprioception corrupter
        def prop_corrupter(inp):
            inp_c = np.array(inp)

            # Generate noise
            pos_noise = self.prop_noise_std[0] * np.random.randn(3)
            rot_noise = self.prop_noise_std[1] * np.random.randn(3)

            # Apply noise to position measurement
            inp_c[:3] += pos_noise[:3]

            # Apply noise to rotation measurement
            quat = inp_c[3:7]
            inp_c[3:7] = T.quat_multiply(quat, T.mat2quat(T.euler2mat(rot_noise[:3])))

            return inp_c

        sensors = [ft_all_denoised, ft_all, prop_all]
        corrupters = [None, ft_corrupter, prop_corrupter]
        names = [s.__name__ for s in sensors]

        for name, s, c in zip(names, sensors, corrupters):
            observables[name] = Observable(
                name=name,
                sensor=s,
                corrupter=c,
                sampling_rate=self.control_freq,
            )

        return observables


