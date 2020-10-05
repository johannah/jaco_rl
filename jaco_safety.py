

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import dm_env
from dm_env import specs
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control import robot
from IPython import embed
from copy import deepcopy
import numpy as np
import time 
# the kinova jaco2 ros exposes the joint state at ~52Hz
# CONTROL_TIMESTEP should be long enough that position controller can also reach the destination in mujoco if it is reacher. Tested with .1 CONTROL_TIMESTEP and .1 maximum relative deg step in 7DOF jaco for position controllers (with tuned gain).
_CONTROL_TIMESTEP = .1
_POSITION_EPISODE_TIME_LIMIT = 2 # for position controller - these seconds divided by thethe timestep specified in the xml file (.02) is the number of steps per episode
_LONG_POSITION_EPISODE_TIME_LIMIT = 20 # for position controller - we the time limit doesnt really make sense
_LONG_EPISODE_TIME_LIMIT = 20
_SHORT_EPISODE_TIME_LIMIT = 10
_TINY_EPISODE_TIME_LIMIT = 5
_BIG_TARGET = .05
_SMALL_TARGET = .015
# 7DOF Jaco2
#D1 Base to shoulder 0.2755
#D2 First half upper arm length 0.2050
#D3 Second half upper arm length 0.2050
#D4 Forearm length (elbow to wrist) 0.2073
#D5 First wrist length 0.1038
#D6 Second wrist length 0.1038
#D7 Wrist to center of the hand 0.1600
#e2 Joint 3-4 lateral offset 0.0098

# Params for Denavit-Hartenberg Reference Frame Layout (DH)
DH_lengths =  {'D1':0.2755, 'D2':0.2050, 
                    'D3':0.2050, 'D4':0.2073,
                    'D5':0.1038, 'D6':0.1038, 
                    'D7':0.1600, 'e2':0.0098}

# DH transform from joint angle to XYZ from kinova robotics ros code
DH_a = (0, 0, 0, 0, 0, 0, 0)
DH_d = (-DH_lengths['D1'], 
         0, 
         -(DH_lengths['D2']+DH_lengths['D3']), 
         -DH_lengths['e2'], 
         -(DH_lengths['D4']+DH_lengths['D5']), 
         0, 
         -(DH_lengths['D6']+DH_lengths['D7']))

DH_alpha = (np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi)
DH_theta_sign = (1, 1, 1, 1, 1, 1, 1)
DH_theta_offset = (np.pi,0.0, 0.0, 0.0, 0.0,0.0,np.pi/2.0)


# size of target in meters
SUITE = containers.TaggedTasks()

def DHtransformEL(d,theta,a,alpha):
    T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha),a*np.cos(theta)],
                  [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha),a*np.sin(theta)],
                  [0.0, np.sin(alpha), np.cos(alpha),d],
                  [0.0,0.0,0.0,1.0]])
    return T

def trim_and_check_pose_safety(position, fence):
    """
    take in a position list [x,y,z] and ensure it doesn't violate the defined fence
    """
    hit = False
    safe_position = []
    for ind, dim in enumerate(['x','y','z']):
        if max(fence[dim]) < position[ind]:
            out = max(fence[dim])
            hit = True
            print('hit max: req {} is more than fence {}'.format(position[ind], max(fence[dim])))
        elif position[ind] < min(fence[dim]):
            out = min(fence[dim])
            hit = True
            print('hit min: req {} is less than fence {}'.format(position[ind], min(fence[dim])))
        else:
            out = position[ind]
        safe_position.append(out)
    return safe_position, hit

class JacoSafety(base.Task):
    """A Bring `Task`: bring the prop to the target."""

    def __init__(self, safety_physics, random=None, action_penalty=True, relative_rad_max=.1, fence = {'x':(-1,1),'y':(-1,1),'z':(-1.2,1.2)}, degrees_of_freedom=7):

        """Initialize an instance of `Jaco`.
        Args:
         random: int seed number for random seed

         fully_observable: A `bool` not yet used
         action_penalty: bool impose a penalty for actions
         relative_step: bool indicates that actions are relative to previous step. Set to True for sim2real as we need to ensure that the actions trained in dm_control can be completed within the control step as they are in the real blocking ros system.
         relative_rad_max: float indicating the maximum relative step. Tested 7dof robot with .2 control_timestep and relative position of max 0.1 rads
         fence: dict with {'x':(min,max), 'y':(min,max), 'z':(min,max)} indicating a virtual cartesian fence. We impose a penalty for extreme joints which exit the virtual fence in dm_control and impose a hard limit on the real robot.
         degrees_of_freedom: int indicating the number of joints to be controlled
         extreme_joints: list of joints (starting with joint 1) to consider for imposing fence violations in dm_control. For 7dof Jaco, this should be [4,6,7]  out of joints (1,2,3,4,5,6,7).
         target_size: float indicating the size of target in reaching tasks
         target_type: string indicating if we should calculate a 'random' or 'fixed' position for the target at reset. If fixed, will used fixed_target_position
         fixed_target_position: list indicating x,y,z center of target in cartesian space
 
            location.
          relative_step: bool whether input action should be relative to current position or absolute
          relative_rad_max: float limit radian step to min/max of this value
          random: Optional,  an integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self.safety_physics = safety_physics
        self.target_type = target_type
        self.fixed_target_position = self.target_position = np.array(fixed_target_position)
        self.max_target_distance_from_tool = 1.0
        self.max_target_distance_from_base = 1.0 # dont extend past arm distance
        self.relative_rad_max = relative_rad_max
        self.DOF = degrees_of_freedom
        self.fence = fence
        self.use_action_penalty = bool(action_penalty)
        # ~.13 m from tool pose to fingertip
        # seems like it odesnt really train if offset < .1, def works at .15
        # radii = physics.named.model.geom_size[['target', 'finger'], 0].sum()
        self.radii = self.target_size + .15
        self.random_state = np.random.RandomState(random)
        self._fully_observable = fully_observable
        # TODO are open/close opposite on robot??
        self.opened_hand_position = np.zeros(6)
        self.closed_hand_position = np.array([1.1,0.1,1.1,0.1,1.1,0.1])

        # find target min/max using fence and considering table obstacle and arm reach
        # TODO Hard limits - should be made vars
        self.target_minx = max([min(self.fence['x'])]+[-.8])
        self.target_maxx = min([max(self.fence['x'])]+[.8])
        self.target_miny = max([min(self.fence['y'])]+[-.8])
        self.target_maxy = min([max(self.fence['y'])]+[.8])
        self.target_minz = max([min(self.fence['z'])]+[0.1])
        self.target_maxz = min([max(self.fence['z'])]+[.8])
        print('Jaco received virtual fence of:', self.fence)
        print('limiting target to x:({},{}), y:({},{}), z:({},{})'.format(
                               self.target_minx, self.target_maxx,
                               self.target_miny, self.target_maxy,
                               self.target_minz, self.target_maxz))
        self.sky_joint_angles = np.array([-6.27,3.27,5.17,3.24,0.234,3.54,3.14,
                                  1.1,0.0,1.1,0.0,1.1,0.])
        self.out_joint_angles = np.array([-6.27,1,5.17,3.24,0.234,3.54,3.14,
                                  1.1,0.0,1.1,0.0,1.1,0.])
 
        ## approx loc on home on real 7dof jaco2 robot
        self.sleep_joint_angles = np.array([4.71,  # 270 deg
                                  2.61,   # 150
                                  0,      # 0
                                  .5,     # 28
                                  6.28,   # 360
                                  3.7,    # 212
                                  3.14,   # 180
                                  1.1,0.1,1.1,0.1,1.1,0.1])
        # true home on the robot has the fingers open
        self.home_joint_angles = np.array([4.92,    # 283 deg
                                  2.839,   # 162.709854126
                                  0,       # 0 
                                  .758,    # 43.43
                                  4.6366,  # 265.66
                                  4.493,   # 257.47
                                  5.0249,  # 287.9
                                  1.1,0.1,1.1,0.1,1.1,0.1])
        robot_name = 'js7s300' 
        robot_model = robot_name[:2]
        assert robot_model == 'j2'
        # only tested with 7dof, though 6dof should work with tweaks
        self.n_major_actuators = int(robot_name[3:4])
        assert self.n_major_actuators == 7
        # only tested with s3 hand
        hand_type = robot_name[4:6]
        assert hand_type == 's3'
        if hand_type == 's3':
            # 3 in new dm, 6 in robot
            self.n_hand_actuators = 3
        self.n_actuators = self.n_major_actuators + self.n_hand_actuators
        # TODO - get names automatically - need to exclude base / objects in scene
        self.body_parts = ['b_1', 'b_2', 'b_3', 'b_4', 'b_5', 'b_6', 'b_7',
                           'b_finger_1', 'b_finger_tip_1',
                           'b_finger_2', 'b_finger_tip_2',
                           'b_finger_3', 'b_finger_tip_3']

        self.body_ids = [self.model.name2id(bp, 'body') for bp in self.body_parts]
        self.n_actuators = self.n_major_actuators + self.n_hand_actuators
        # TODO - get names automatically - need to exclude base / objects in scene
        self.body_parts = ['b_1', 'b_2', 'b_3', 'b_4', 'b_5', 'b_6', 'b_7', 
                           'b_finger_1', 'b_finger_tip_1', 
                           'b_finger_2', 'b_finger_tip_2', 
                           'b_finger_3', 'b_finger_tip_3']

        self.body_ids = [self.model.name2id(bp, 'body') for bp in self.body_parts]
        # TODO might want to expand in future - this will only detect body 2 body collisions
        self.specific_collision_geom_ids = self.body_ids
        # allow agent to control this many of the joints (starting with 0)
        self.random_state = np.random.RandomState(seed)
        self.actuated_joint_names = self.named.data.qpos.axes.row.names

        super(Jaco, self).__init__()

    def get_position_angles_by_name(self, position_name='home'):
        if position_name == 'home':
            angles = self.home_joint_angles
        elif position_name == 'sky':
            angles = self.sky_joint_angles
        elif position_name == 'out':
            angles = self.out_joint_angles
        elif position_name == 'random':
             angles = self.find_random_joint_angles()
        else:
            raise NotImplementedError
        return angles

    def check_action_safety(self, action):
        self.safety_physics.step(action)
        self.count_safety_violations()

#    def find_random_joint_angles(self, max_trys=10000):
#        safe = False
#        st = time.time()
#        # TODO this doesn't work in new dm
#        embed()
#        bounds = self.safety_physics.action_spec()
#        # clip rotations to one revolution
#        min_bounds = bounds.minimum.clip(-np.pi*2, np.pi*2)
#        max_bounds = bounds.maximum.clip(-np.pi*2, np.pi*2)
#        trys = 0
#        while not safe and trys < max_trys:
#            random_angles = self.random_state.uniform(min_bounds, max_bounds, len(min_bounds))
#            trys+=1
#            if not self.count_safety_violations(random_angles):
#                et = time.time()
#                print('took %s seconds and %s trys to find random position'%((et-st), trys))
#                return random_angles
#        
#        print('unable to find safe random joints after {} trys'.format(trys))
#        return self.home_joint_angles

    def set_joint_angles(self, body_angles):
        # fingers are always last in xml - joint angles are for major joints to least major
        self.safety_physics.named.data.qpos[self.actuated_joint_names[:len(body_angles)]] = body_angles

    def count_safety_violations(self):
        violations = 0
        self.safety_physics.after_reset()
        penetrating = self.safety_physics.data.ncon
        if penetrating > 0: 
            for contact in self.safety_physics.data.contact:
                if contact.geom1 in self.safety_physics.specific_collision_geom_ids and contact.geom2 in self.safety_physics.specific_collision_geom_ids:
                    contact_name1 = self.safety_physics.body_parts[self.safety_physics.body_ids.index(contact.geom1)]
                    contact_name2 = self.safety_physics.body_parts[self.safety_physics.body_ids.index(contact.geom2)]
                    print("{} collided with {}".format(contact_name1, contact_name2))
                    violations += 1
        positions = self.safety_physics.named.data.xpos[self.safety_physics.body_parts]
        for position in positions:
            violations += self.is_violating_fence(position)
        return violations
            
    def is_violating_fence(self, position):
        violations = 0
        assert len(position) == 3
        for ind, var in enumerate(['x','y','z']): 
            vals = position[ind]
            if vals < min(self.fence[var]):
                print('fence min', var, vals, min(self.fence[var]))
                violations += 1
            if vals > max(self.fence[var]):
                print('fence max', var, vals, max(self.fence[var]))
                violations += 1
        return violations
 

