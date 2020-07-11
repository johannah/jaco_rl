-- jaco_rl
off by one in first iteration? 
plot all eval
bootstrap / prior
load exp type etc if load_model
save .py files
plot force, velocity, 
reload some parameters - including fence / exp name for testing
add capsule into frame
change load_path.pt to load_base_path
add steps to replay_buffer

-- dm_control:
add target position return in to env creation
how much does the fence slow down computation? 
should we turn fence off and on?
figure out proper mujoco tool pose
penalize collisions with itself
how does extreme joints and xpos differ? I'm not sure yet if this is a problem for us, but I saw this line in the kinova repo: 
https://github.com/Kinovarobotics/kinova-ros/blob/d2876ad92c075dda8d8ebd7663712046641a30ea/kinova_driver/src/kinova_arm_kinematics.cpp#L216
// IMPORTANT!!! In the robot DSP chip, the classical D-H parameters are used to define the robot. Therefore, the frame definition is different comparing with the frames in URDF model.


-- ros_interface
ctrl-c isnt working server
switch to using json for msg passing

-- action sequence 
load data action sequence data
sequence of actions and reconstruct
it needs the location of the current position of the joints
it may be helpful to also predict forces
are the clustered sequences useful to agents/humans for building more complex tasks
can we iteratively adapt the clusters - 

# NOTES
- don't add capsules to the xml file ! they mess up physics
- tried adding joint force penalty, but they don't really scale well and it encouraged the agent to just use the 0th joint to spin since it always has relatively little force


# PAPERS

discrete/continuous
https://arxiv.org/pdf/1705.05035.pdf

