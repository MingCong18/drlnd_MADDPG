# main function that sets up environments
# perform training loop

import os
import torch
import numpy as np 
from tensorboardX import SummaryWriter

import envs
from buffer import ReplayBuffer
from maddpg import MADDPG

from utilities import transpose_list, transpose_to_tensor


def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)



def main():
seeding()
# number of parallel agents
parallel_envs = 4
# number of training episodes.
# change this to higher number to experiment. say 30000.
number_of_episodes = 1000
episode_length = 80
batchsize = 1000
# how many episodes to save policy and gif
save_interval = 1000
t = 0


# amplitude of OU noise
# this slowly decreases to 0
noise = 2
noise_reduction = 0.9999

# how many episodes before update
episode_per_update = 2 * parallel_envs

log_path = os.getcwd() + '/log'
model_dir = os.getcwd() + '/model_dir'

os.makedirs(model_dir, exist_ok=True)

torch.set_num_threads(parallel_envs)
env = envs.make_parallel_env(parallel_envs)

# keep 5000 episodes worth of replay
buffer = ReplayBuffer(int(5000 * episode_length))

# initialize policy and critic
maddpg = MADDPG()
logger = SummaryWriter(log_dir = log_path)
agent0_reward = []
agent1_reward = []
agent2_reward = []

# training loop
# show progressbar
import progressbar as pb
widget = ['episode: ', pb.Counter(), '/', str(number_of_episodes), ' ',
		  pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ' ]

timer = pb.ProgressBar(widgets=widget, maxval=number_of_episodes).start()

# use keep_awake to keep workspace from disconnecting
# for episode in keep_awake(range(0, number_of_episodes, parallel_envs)):
for episode in range(0, number_of_episodes, parallel_envs):

	timer.update(episode)

	reward_this_episode = np.zeros((parallel_envs, 3))
	all_obs = env.reset()
	obs, obs_full = transpose_list(all_obs)

	# for calculating rewards for this particular episode - addition of all time steps

	# save info or not
	save_info = ((episode) % save_interval < parallel_envs or episode==number_of_episodes-parallel_envs)
	frames = []
	tmax = 0

	if save_info:
		frames.append(env.render('rgb_array'))




















