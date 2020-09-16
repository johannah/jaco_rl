import torch
import numpy as np
from IPython import embed
from torch.autograd import Variable
import time
from copy import deepcopy
random_state = np.random.RandomState(222)
# 7DOF Jaco2
#D1 Base to shoulder 0.2755
#D2 First half upper arm length 0.2050
#D3 Second half upper arm length 0.2050
#D4 Forearm length (elbow to wrist) 0.2073
#D5 First wrist length 0.1038
#D6 Second wrist length 0.1038
#D7 Wrist to center of the hand 0.1600
#e2 Joint 3-4 lateral offset 0.0098

        
def DHtransform(d,theta,a,alpha):
    T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha),a*np.cos(theta)],
                      [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha),a*np.sin(theta)],
                      [0.0, np.sin(alpha), np.cos(alpha),d],
                      [0.0,0.0,0.0,1.0]])
    return T

def find_joint_coordinate_extremes(dh_dict, major_joint_angles):  
    """calculate xyz positions for joints form cartesian extremes
    major_joint_angles: ordered list of joint angles in radians (len 7 for 7DOF arm)"""
    extreme_xyz = []
    # for 7DOF robot - transform first!
    Tall = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=np.float)
    for i, angle in enumerate(major_joint_angles):
        DH_theta = dh_dict['DH_theta_sign'][i]*angle + dh_dict['DH_theta_offset'][i]
        T = DHtransform(dh_dict['DH_d'][i], DH_theta, dh_dict['DH_a'][i], dh_dict['DH_alpha'][i])
        Tall = np.dot(Tall, T)
        #if i+1 in self.extreme_joints:
        extreme_xyz.append([Tall[0,3], Tall[1,3], Tall[2,3]])
    extremes = np.array(extreme_xyz)
    return extremes

def torch_DHtransform(d,theta,a,alpha):
    # need to add DH params individually to get grad thru - creating new float tensor didnt work
    T = torch.zeros((4,4), device=alpha.device)
    T[0,0] = T[0,0] +  torch.cos(theta)
    T[0,1] = T[0,1] + -torch.sin(theta)*torch.cos(alpha)
    T[0,2] = T[0,2] +  torch.sin(theta)*torch.sin(alpha)
    T[0,3] = T[0,3] +  a*torch.cos(theta)
    T[1,0] = T[1,0] +  torch.sin(theta)
    T[1,1] = T[1,1] +   torch.cos(theta)*torch.cos(alpha)
    T[1,2] = T[1,2] +   -torch.cos(theta)*torch.sin(alpha)
    T[1,3] = T[1,3] +  a*torch.sin(theta)
    #T[2,0] =  0.0
    T[2,1] = T[2,1] +  torch.sin(alpha)
    T[2,2] = T[2,2] +   torch.cos(alpha)
    T[2,3] = T[2,3] +  d
    #T[3,0] = T[3,0] +  0.0
    #T[3,1] = T[3,1] +  0.0
    #T[3,2] = T[3,2] +  0.0
    T[3,3] = T[3,3] +  1.0
    ###- .0002 cpu, .005 gpu
    return T 


def get_torch_attributes(np_attribute_dict, device='cpu'):
    pt_attribute_dict = {}
    for key, item in np_attribute_dict.items():
        pt_attribute_dict[key] = torch.FloatTensor(item).to(device)
    return pt_attribute_dict

class torchDHtransformer():
    def __init__(self, np_attribute_dict, device):
        self.device = device
        self.tdh_dict = get_torch_attributes(np_attribute_dict, device=self.device)
        
        self.base_tTall = Variable(torch.FloatTensor([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]).to(device), requires_grad=True)


    def find_joint_coordinate_extremes(self, tmajor_joint_angles, base_tTall, return_T=False, angle_dh_indexes=[]):
        """Pytorch version calculate xyz positions for joints form cartesian extremes
        major_joint_angles: ordered list of joint angles in radians (len 7 for 7DOF arm)"""
        device = tmajor_joint_angles.device
        extreme_xyz = torch.zeros((tmajor_joint_angles.shape[0],3)).to(device)
        if len(angle_dh_indexes) == 0:
            # assume we start indexing dh at 0
            angle_dh_indexes = [x for x in range(len(tmajor_joint_angles))]

        i = angle_dh_indexes[0]
        outi = 0
        DH_theta = self.tdh_dict['DH_theta_sign'][i]*tmajor_joint_angles[outi] + self.tdh_dict['DH_theta_offset'][i]
        T = torch_DHtransform(self.tdh_dict['DH_d'][i], DH_theta, self.tdh_dict['DH_a'][i], self.tdh_dict['DH_alpha'][i])
        tTall = torch.matmul(base_tTall,T)
        extreme_xyz[outi,0] = extreme_xyz[outi,0] + tTall[0,3]
        extreme_xyz[outi,1] = extreme_xyz[outi,1] + tTall[1,3] 
        extreme_xyz[outi,2] = extreme_xyz[outi,2] + tTall[2,3]
      
       
        for outi, i in enumerate(angle_dh_indexes[1:]):
            outi += 1
            DH_theta = self.tdh_dict['DH_theta_sign'][i]*tmajor_joint_angles[outi] + self.tdh_dict['DH_theta_offset'][i]
            T = torch_DHtransform(self.tdh_dict['DH_d'][i], DH_theta, self.tdh_dict['DH_a'][i], self.tdh_dict['DH_alpha'][i])
            tTall = torch.matmul(tTall,T)
            extreme_xyz[outi,0] = extreme_xyz[outi,0] + tTall[0,3]
            extreme_xyz[outi,1] = extreme_xyz[outi,1] + tTall[1,3] 
            extreme_xyz[outi,2] = extreme_xyz[outi,2] + tTall[2,3]
        if return_T:
            return extreme_xyz, tTall
        else:
            return extreme_xyz

#    def find_joint_coordinate_extremes(self, tmajor_joint_angles, base_tTall, return_T=False):
#        """Pytorch version calculate xyz positions for joints form cartesian extremes
#        major_joint_angles: ordered list of joint angles in radians (len 7 for 7DOF arm)"""
#        device = tmajor_joint_angles.device
#        extreme_xyz = torch.zeros((tmajor_joint_angles.shape[0],3)).to(device)
#        i = 0
#        DH_theta = self.tdh_dict['DH_theta_sign'][i]*tmajor_joint_angles[i] + self.tdh_dict['DH_theta_offset'][i]
#        T = self.torch_DHtransform(self.tdh_dict['DH_d'][i], DH_theta, self.tdh_dict['DH_a'][i], self.tdh_dict['DH_alpha'][i])
#        tTall = torch.matmul(base_tTall,T)
#        extreme_xyz[i,0] = extreme_xyz[i,0] + tTall[0,3]
#        extreme_xyz[i,1] = extreme_xyz[i,1] + tTall[1,3] 
#        extreme_xyz[i,2] = extreme_xyz[i,2] + tTall[2,3]
#       
#        for i in range(1, len(tmajor_joint_angles)):
#            DH_theta = self.tdh_dict['DH_theta_sign'][i]*tmajor_joint_angles[i] + self.tdh_dict['DH_theta_offset'][i]
#            T = self.torch_DHtransform(self.tdh_dict['DH_d'][i], DH_theta, self.tdh_dict['DH_a'][i], self.tdh_dict['DH_alpha'][i])
#            tTall = torch.matmul(tTall,T)
#            extreme_xyz[i,0] = extreme_xyz[i,0] + tTall[0,3]
#            extreme_xyz[i,1] = extreme_xyz[i,1] + tTall[1,3] 
#            extreme_xyz[i,2] = extreme_xyz[i,2] + tTall[2,3]
#        if return_T:
#            return extreme_xyz, tTall
#        else:
#            return extreme_xyz

def test_torch_extremes(states, device):
    torch_DH_attributes_jaco27DOF = get_torch_attributes(DH_attributes_jaco27DOF, device=device)
    extremes = np.array([find_joint_coordinate_extremes(DH_attributes_jaco27DOF, states[x]) for x in range(states.shape[0])])

    torch_dh = torchDHtransformer(DH_attributes_jaco27DOF, device)
    torch_states = Variable(torch.FloatTensor(states).to(device), requires_grad=True)
    torch_extremes = torch.stack([torch_dh.find_joint_coordinate_extremes(torch_states[x], torch_dh.base_tTall) for x in range(states.shape[0])])
    np_torch_extremes = torch_extremes.detach().cpu().numpy()
    diff = (extremes-np_torch_extremes)**2
    assert diff.max() < .001
 
def test_torch_partial_extremes(states, device):
    torch_DH_attributes_jaco27DOF = get_torch_attributes(DH_attributes_jaco27DOF, device=device)
    #extremes = np.array([find_joint_coordinate_extremes(DH_attributes_jaco27DOF, states[x]) for x in range(states.shape[0])])

    for x in range(states.shape[0]):
        s1 = deepcopy(states[x])
        s1a = deepcopy(states[x])
        #s1a[5] = 11
        torch_dh = torchDHtransformer(DH_attributes_jaco27DOF, device)
        ts1 = Variable(torch.FloatTensor(s1).to(device), requires_grad=True)
        ts1a = Variable(torch.FloatTensor(s1a).to(device), requires_grad=True)
        ts1a[5] = random_state.randint(states[5].min(), states[5].max())
        # angle of joint 7 does not matter for cartesian position
        ts1a[6] = random_state.randint(states[6].min(), states[6].max())

        e, t1o = torch_dh.find_joint_coordinate_extremes(ts1, torch_dh.base_tTall, return_T=True)
        ea, t1ao = torch_dh.find_joint_coordinate_extremes(ts1a, torch_dh.base_tTall, return_T=True)
        
        e1, t1 = torch_dh.find_joint_coordinate_extremes(ts1[:5], torch_dh.base_tTall, return_T=True)
        e1a, t1a = torch_dh.find_joint_coordinate_extremes(ts1a[:5], torch_dh.base_tTall, return_T=True)
        e2, t2 = torch_dh.find_joint_coordinate_extremes(ts1[5:], t1, return_T=True, angle_dh_indexes=[5,6])
        e2a, t2a = torch_dh.find_joint_coordinate_extremes(ts1a[5:], t1a, return_T=True, angle_dh_indexes=[5,6])
        o = e.detach().cpu().numpy() 
        oa = ea.detach().cpu().numpy() 
        os = torch.cat((e1,e2)).detach().cpu().numpy()
        osa = torch.cat((e1a,e2a)).detach().cpu().numpy()
        assert (o == os).sum() == (o.shape[0]*o.shape[1])
        assert (oa == osa).sum() == (o.shape[0]*o.shape[1])
    #np_torch_extremes = torch_extremes.detach().cpu().numpy()
    #diff = (extremes-np_torch_extremes)**2
    #assert diff.max() < .001
 
# Params for Denavit-Hartenberg Reference Frame Layout (DH)
jaco27DOF_DH_lengths = {'D1':0.2755, 'D2':0.2050, 
               'D3':0.2050, 'D4':0.2073,
               'D5':0.1038, 'D6':0.1038, 
               'D7':0.1600, 'e2':0.0098}
 
DH_attributes_jaco27DOF = {
          'DH_a':[0, 0, 0, 0, 0, 0, 0],
           'DH_alpha':[np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi],
           'DH_theta_sign':[1, 1, 1, 1, 1, 1, 1],
           'DH_theta_offset':[np.pi,0.0, 0.0, 0.0, 0.0,0.0,np.pi/2.0],
           'DH_d':(-jaco27DOF_DH_lengths['D1'], 
                    0, 
                    -(jaco27DOF_DH_lengths['D2']+jaco27DOF_DH_lengths['D3']), 
                    -jaco27DOF_DH_lengths['e2'], 
                    -(jaco27DOF_DH_lengths['D4']+jaco27DOF_DH_lengths['D5']), 
                    0, 
                    -(jaco27DOF_DH_lengths['D6']+jaco27DOF_DH_lengths['D7']))
           }
 
cuda0_base_tTall = Variable(torch.FloatTensor([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]).to('cuda:0'), requires_grad=True)
cuda1_base_tTall = Variable(torch.FloatTensor([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]).to('cuda:1'), requires_grad=True)



if __name__ == '__main__':
    from replay_buffer import ReplayBuffer, compress_frame
    import pickle
    import os
    #device = 'cuda:0'
    load_dir = 'results/sep8randomClose02/'
    eval_dir = 'sep8randomClose_TD3_jaco_00000_0001140000_eval_CAM-1_S_S1140000_E0010/'
    sim_replay_file = 'test_TD3_jaco_00000_0001140000_eval_CAM-1_S_S1140000_E0010.epkl'
    sim_replay_path = os.path.join(load_dir, eval_dir, sim_replay_file)
    sbuffer = pickle.load(open(sim_replay_path, 'rb'))
    
    states = sbuffer.states[:sbuffer.size,3:3+7] 

    st = time.time()
    #test_torch_extremes(states, device='cuda:1')
    #print('gpu test took', time.time()-st)
    #st = time.time()
    #test_torch_extremes(states, device='cuda:1')
    #print('gpu test took', time.time()-st)
    st = time.time()
    #test_torch_extremes(states, device='cpu')
    #print('cpu test took', time.time()-st)
    test_torch_partial_extremes(states, device='cpu')



#def trim_and_check_pose_safety(position, fence):
#    """
#    take in a position list [x,y,z] and ensure it doesn't violate the defined fence
#    """
#    hit = False
#    safe_position = []
#    for ind, dim in enumerate(['x','y','z']):
#        if max(fence[dim]) < position[ind]:
#            out = max(fence[dim])
#            hit = True
#            print('hit max: req {} is more than fence {}'.format(position[ind], max(fence[dim])))
#        elif position[ind] < min(fence[dim]):
#            out = min(fence[dim])
#            hit = True
#            print('hit min: req {} is less than fence {}'.format(position[ind], min(fence[dim])))
#        else:
#            out = position[ind]
#        safe_position.append(out)
#    return safe_position, hit
#
#class JacoPhysics():
#    """Physics with additional features for the Planar Manipulator domain."""
#
#    def initialize(self, robot_name='j2s7s300', degrees_of_freedom=7, control_type='position'):
#        # only compatible with j2
#        robot_model = robot_name[:2]
#        assert robot_model == 'j2'
#        # only tested with 7dof, though 6dof should work with tweaks 
#        self.n_major_actuators = int(robot_name[3:4])
#        assert self.n_major_actuators == 7 
#        # only tested with s3 hand
#        hand_type = robot_name[4:6]
#        assert hand_type == 's3'
#        if hand_type == 's3':
#            self.n_hand_actuators = 6 
#        self.n_actuators = self.n_major_actuators + self.n_hand_actuators
#        # TODO - get names automatically - need to exclude base / objects in scene
#        self.body_parts = ['b_1', 'b_2', 'b_3', 'b_4', 'b_5', 'b_6', 'b_7', 
#                           'b_finger_1', 'b_finger_tip_1', 
#                           'b_finger_2', 'b_finger_tip_2', 
#                           'b_finger_3', 'b_finger_tip_3']
#
#        self.body_ids = [self.model.name2id(bp, 'body') for bp in self.body_parts]
#        # TODO might want to expand in future - this will only detect body 2 body collisions
#        self.specific_collision_geom_ids = self.body_ids
#        # only position tested
#        self.control_type = control_type
#        self.fence = fence
#        self.type = 'mujoco'
#        # allow agent to control this many of the joints (starting with 0)
#        self.DOF = degrees_of_freedom
#        self.random_state = np.random.RandomState(seed)
#        self.actuated_joint_names = self.named.data.qpos.axes.row.names
#        self.sub_step_limit = 10000
#        self.error_step_complete = 0.01
#
#        self.opened_hand_position = np.zeros(6)
#        self.closed_hand_position = np.array([1.1,0.1,1.1,0.1,1.1,0.1])
#
#        # find target min/max using fence and considering table obstacle and arm reach
#        # TODO Hard limits - should be made vars
#        self.target_minx = max([min(self.fence['x'])]+[-.8])
#        self.target_maxx = min([max(self.fence['x'])]+[.8])
#        self.target_miny = max([min(self.fence['y'])]+[-.8])
#        self.target_maxy = min([max(self.fence['y'])]+[.8])
#        self.target_minz = max([min(self.fence['z'])]+[0.1])
#        self.target_maxz = min([max(self.fence['z'])]+[.8])
#        print('Jaco received virtual fence of:', self.fence)
#        print('limiting target to x:({},{}), y:({},{}), z:({},{})'.format(
#                               self.target_minx, self.target_maxx,
#                               self.target_miny, self.target_maxy,
#                               self.target_minz, self.target_maxz))
#        self.sky_joint_angles = np.array([-6.27,3.27,5.17,3.24,0.234,3.54,3.14,
#                                  1.1,0.0,1.1,0.0,1.1,0.])
#        self.out_joint_angles = np.array([-6.27,1,5.17,3.24,0.234,3.54,3.14,
#                                  1.1,0.0,1.1,0.0,1.1,0.])
# 
#        ## approx loc on home on real 7dof jaco2 robot
#        self.sleep_joint_angles = np.array([4.71,  # 270 deg
#                                  2.61,   # 150
#                                  0,      # 0
#                                  .5,     # 28
#                                  6.28,   # 360
#                                  3.7,    # 212
#                                  3.14,   # 180
#                                  1.1,0.1,1.1,0.1,1.1,0.1])
#        # true home on the robot has the fingers open
#        self.home_joint_angles = np.array([4.92,    # 283 deg
#                                  2.839,   # 162.709854126
#                                  0,       # 0 
#                                  .758,    # 43.43
#                                  4.6366,  # 265.66
#                                  4.493,   # 257.47
#                                  5.0249,  # 287.9
#                                  1.1,0.1,1.1,0.1,1.1,0.1])
# 
#
#       
#    def get_position_angles_by_name(self, position_name='home'):
#        if position_name == 'home':
#            angles = self.home_joint_angles
#        elif position_name == 'sky':
#            angles = self.sky_joint_angles
#        elif position_name == 'out':
#            angles = self.out_joint_angles
#        elif position_name == 'random':
#             angles = self.find_random_joint_angles()
#        else:
#            raise NotImplementedError
#        return angles
#
#    def find_random_joint_angles(self, max_trys=10000):
#        safe = False
#        st = time.time()
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
#
#
#    def count_safety_violations(self, joint_angles):
#        violations = 0
#        self.safety_physics.set_joint_angles(joint_angles)
#        self.safety_physics.after_reset()
#        penetrating = self.safety_physics.data.ncon
#        if penetrating > 0: 
#            for contact in self.safety_physics.data.contact:
#                if contact.geom1 in self.safety_physics.specific_collision_geom_ids and contact.geom2 in self.safety_physics.specific_collision_geom_ids:
#                    contact_name1 = self.safety_physics.body_parts[self.safety_physics.body_ids.index(contact.geom1)]
#                    contact_name2 = self.safety_physics.body_parts[self.safety_physics.body_ids.index(contact.geom2)]
#                    print("{} collided with {}".format(contact_name1, contact_name2))
#                    violations += 1
#        positions = self.safety_physics.named.data.xpos[self.safety_physics.body_parts]
#        for position in positions:
#            violations += self.is_violating_fence(position)
#        return violations
#            
#    def is_violating_fence(self, position):
#        violations = 0
#        assert len(position) == 3
#        for ind, var in enumerate(['x','y','z']): 
#            vals = position[ind]
#            if vals < min(self.fence[var]):
#                print('fence min', var, vals, min(self.fence[var]))
#                violations += 1
#            if vals > max(self.fence[var]):
#                print('fence max', var, vals, max(self.fence[var]))
#                violations += 1
#        return violations
# 
#    def make_target(self, target_type='random', limit_distance_from=[0,0,0]):
#        assert target_type in ['random', 'fixed']
#        if target_type == 'random':
#            # TODO - should make sure it doesn't collide with the current robot position
#            # limit distance from tool pose to within max_distance meters of the tool
#            base_distance = self.max_target_distance_from_base + 10
#            attempts = 0
#            px, py, pz = limit_distance_from
#            tx, ty, tz = 1000, 1000, 1000
#            target_position = np.array([tx,ty,tz])
#            while base_distance > self.max_target_distance_from_base or self.is_violating_fence(target_position):
#                r = self.random_state.uniform(0, self.max_target_distance_from_tool)
#                theta = self.random_state.uniform(0, np.pi)
#                phi = self.random_state.uniform(0, 2*np.pi)
#                tx = px + r*np.sin(theta)*np.cos(phi)
#                ty = py + r*np.sin(theta)*np.sin(phi)
#                tz = pz + r*np.cos(theta)
#                base_distance = np.sqrt(tx**2 + ty**2 + tz**2)
#                target_position = np.array([tx,ty,tz])
#                attempts += 1
#                print('attempt {} at finding target {}'.format(attempts, target_position))
#                print('basedistance {}'.format(base_distance))
# 
#        elif target_type == 'fixed':
#            target_position = self.fixed_target_position
#        else:
#            raise ValueError; print('unknown target_type: fixed or random are required but was set to {}'.format(target_type))
#        print('**setting new target position of:{}**'.format(target_position))
#        return target_position
#
#
#    def initialize_episode(self, physics):
#        """Sets the state of the environment at the start of each episode."""
#        print('initialize episode')
#        physics.set_joint_angles(self.get_position_angles_by_name(self.start_position))
#        self.after_step(physics)
#        print('starting episode with joint angles: {}'.format(self.joint_angles))
#        print('starting episode with tool position: {}'.format(self.tool_position))
#        # tool position and joint position are updated in after step
#        self.last_tool_position = deepcopy(self.tool_position)
#        self.target_position = self.make_target(self.target_type, self.tool_position)
#        # init vars
#        physics.set_position_of_target(self.target_position, self.target_size)
#        super(Jaco, self).initialize_episode(physics)
#
#    def before_step(self, action, physics):
#        """
#        take in relative or absolute np.array of actions of length self.DOF
#        if relative mode, action will be relative to the current state
#        """
#        # need action to be same shape as number of actuators
#        if self.relative_step:
#            # relative action prone to drift over time
#            relative_action = np.clip(action, -self.relative_rad_max, self.relative_rad_max)
#            use_action = relative_action+self.joint_angles[:len(action)]
#        else:
#            use_action = np.clip(action, self.joint_angles[:len(action)]-self.relative_rad_max, self.joint_angles[:len(action)]+self.relative_rad_max)
#
#        if len(use_action) < physics.n_actuators:
#            use_action = np.hstack((use_action, self.closed_hand_position))
#        super(Jaco, self).before_step(use_action, physics)
#
#    def after_step(self, physics):
#        self.joint_angles = deepcopy(physics.get_joint_angles_radians())
#        # this sets safety physics
#        self.hit_penalty = -self.count_safety_violations(self.joint_angles)
#        print('after step penalty', self.hit_penalty)
#        self.last_tool_position = deepcopy(self.tool_position)
#        self.tool_position = self.safety_physics.named.data.site_xpos['pinchsite']
#        
#    def get_observation(self, physics):
#        """Returns either features or only sensors (to be used with pixels)."""
#        obs = collections.OrderedDict()
#        #print("TOOL POSITION", physics.type, self.tool_position)
#        obs['to_target'] = self.target_position-self.tool_position
#        obs['joint_angles'] = self.joint_angles
#        return obs
#
#    def get_distance(self, position_1, position_2):
#        """Returns the signed distance bt 2 positions"""
#        return np.linalg.norm(position_1-position_2)
#
#    def get_reward(self, physics):
#        """Returns a sparse reward to the agent."""
#        # with policy-gradient algs, the agent shakes a lot w/o some action penalty
#        # Since openai-gym utilises an action penalty on its reacher, I decided to try it here
#        # I've tried a penalty on the joint-position change and on the tool pose change 
#        # The tool pose action penalty resulted in better (smoother) policies with TD3
#        distance = self.get_distance(self.tool_position, self.target_position)
#        action_penalty = -np.square(self.last_tool_position-self.tool_position).sum()
#        return rewards.tolerance(distance, (0, self.radii)) + self.hit_penalty + action_penalty
