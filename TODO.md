-- jaco_rl
off by one in first iteration?  
plot all eval 
bootstrap / prior 
load exp type etc if load_model 
save .py files 
reload some parameters - including fence / exp name for testing 
add force unique acn 
vq issue? 


-- dm_control:
// IMPORTANT!!! In the robot DSP chip, the classical D-H parameters are used to define the robot. Therefore, the frame definition is different comparing with the frames in URDF model.


-- ros_interface
switch to using json for msg passing

-- action sequence 

# NOTES
- tried adding joint force penalty, but they don't really scale well and it encouraged the agent to just use the 0th joint to spin since it always has relatively little force


# PAPERS

discrete/continuous
https://arxiv.org/pdf/1705.05035.pdf

