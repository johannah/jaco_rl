# RL on Jaco Arm in DeepMind Control Suite
This repository provides the tools to train a Jaco arm with dm_control. It depends on a dm_control repository. 
[here](https://github.com/johannah/jaco). 
* The agent implementations are taken from the [Official TD3 repository](https://github.com/sfujim/TD3).


### Example training for reacher:
```
python test.py --domain reacher --task easy --policy TD3 --seed 100 --device 'cuda:0' --exp_name 'reacher_easy'
```

### Example plot and eval previous experiment
``
python test.py --domain reacher --task easy --policy TD3 --seed 101 --load_model 'results/reacher_easy_00'  --eval --state_pixels
```
