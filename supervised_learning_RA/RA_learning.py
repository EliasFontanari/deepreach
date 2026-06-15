import random
from collections import deque

import numpy as np
import torch
from torch import nn
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

state_range = np.array(
	[
		[-4, 4],
		[-4, 4],
		[-4, 4],
		[-1, 1],
		[-1, 1],
		[-1, 1],
		[-1, 1],
		[-3, 3],
		[-3, 3],
		[-3, 3],
		[-3, 3],
		[-3, 3],
		[-3, 3],
	],
	dtype=np.float64,
)

def generate_query_grid(state_slice):
	grid_size = 0.1
	# Create x and y ranges
	x = np.linspace(state_range[0][0], state_range[0][1], int((state_range[0][1] - state_range[0][0]) / grid_size)+2)
	y = np.linspace(state_range[1][0], state_range[1][1], int((state_range[1][1] - state_range[1][0]) / grid_size)+2)

	# Create a meshgrid
	X, Y = np.meshgrid(x, y)

	grid_shape = X.shape

	# Flatten and stack to create (x, y) pairs
	xy_grid = np.stack([X.ravel(), Y.ravel()], axis=1)
	standard_conf_vector = state_slice[2:state_slice.shape[0]]

	query_grid = np.zeros((xy_grid.shape[0],state_slice.shape[0]))
	query_grid[:, :2] = xy_grid
	query_grid[:, 2:] = standard_conf_vector
	return query_grid, grid_shape

def plot_V_XY(grid_values, log_learning):
	fig, ax = plt.subplots()

	plt.grid(True)

	V_flipped = np.flipud(grid_values)
	sns.heatmap(V_flipped, annot=False, cmap=cm.coolwarm_r, ax=ax, vmin= grid_values.min(), vmax=grid_values.max(),
				cbar=True,
				)

	num_x_points = grid_values.shape[1]  # Number of columns
	num_y_points = grid_values.shape[0]  # Number of rows
	
	# Create tick positions at regular intervals
	# Positions are indices in the heatmap (0 to num_points)
	num_ticks_x, num_ticks_y = 5, 3
	x_tick_positions = np.linspace(0, num_x_points , num_ticks_x)
	y_tick_positions = np.linspace(0, num_y_points , num_ticks_y)
	
	# Create tick labels corresponding to actual x, y values
	# Your grid goes from -lim to +lim
	x_tick_labels = np.linspace(state_range[0][0], state_range[0][1], num_ticks_x)
	# Because the data is flipped with flipud, reverse the y labels so they match
	# the original coordinate ordering.
	y_tick_labels = np.linspace(state_range[1][1], state_range[1][0], 3)

	ax.set_xticks(x_tick_positions)
	ax.set_yticks(y_tick_positions)
	ax.set_xticklabels(np.round(x_tick_labels, 2))
	ax.set_yticklabels(np.round(y_tick_labels, 2))

	# ax.set_xticklabels(np.round(x[::x_interval], 2))
	# ax.set_yticklabels(np.round(y[::-y_interval]+0.02, 1))

	level = -2 if log_learning else 0
	x_coords = np.linspace(0, num_x_points - 1, num_x_points)
	y_coords = np.linspace(0, num_y_points - 1, num_y_points)
	X_contour, Y_contour = np.meshgrid(x_coords, y_coords)
	
	contours = ax.contour(X_contour, Y_contour, V_flipped, levels=[level], 
						 colors="black", linestyles='dashed')
	ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")
	plt.xlabel("x")
	plt.ylabel("y")

	# no legend needed for the heatmap
	plt.title(
		f"V_net"
	)

class RAValueFunction(nn.Module):
	def __init__(self, input_dim=1, hidden_dim=256, dropout=0.15):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(p=dropout),   # dropout
			nn.Linear(hidden_dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(p=dropout),   # dropout
			nn.Linear(hidden_dim, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, 1),
			nn.Tanh(),
		)
		# self._init_near_minus_one()

	def _init_near_minus_one(self, target_value=-0.999, weight_scale=1e-2):
		# Initialize near a constant -1 output, but avoid tanh saturation so learning can start.
		for layer in self.net:
			if isinstance(layer, nn.Linear):
				nn.init.normal_(layer.weight, mean=0.0, std=weight_scale)
				nn.init.zeros_(layer.bias)

		# Set last-layer bias so tanh(pre_activation) ~= target_value.
		last_linear = self.net[-2]
		if isinstance(last_linear, nn.Linear):
			bias_value = float(np.arctanh(target_value))
			last_linear.bias.data.fill_(bias_value)

	def forward(self, x):
		return self.net(x)


class TransitionBuffer:
	def __init__(self, queue_len=1001, hindsight=10):
		self.queue_len = queue_len
		self.hindsight = hindsight
		self._buf = deque(maxlen=queue_len)

	def add(self, transition):
		self._buf.append(transition)

	def extend(self, transitions):
		for t in transitions:
			self._buf.append(t)

	def __len__(self):
		return len(self._buf)

	def sample(self, batch_size):
		return random.sample(self._buf, batch_size)


class RALearner:
	def __init__(
		self,
		input_dim=1,
		queue_len=1001,
		batch_size=200,
		hindsight=10,
		gamma=0.999999,
		weight_end=0.0,
		lr=0.0005,
		momentum=0.0,
		device=None,
	):
		self.device = device or (
			"cuda" if torch.cuda.is_available() else "cpu")
		self.model = RAValueFunction(input_dim=input_dim).to(self.device)
		self.target_model = RAValueFunction(input_dim=input_dim).to(self.device)
		self.target_model.load_state_dict(self.model.state_dict())
		self.target_model.eval()
		self.optimizer = torch.optim.SGD(
			self.model.parameters(), lr=lr, momentum=momentum
		)
		self.buffer = TransitionBuffer(
			queue_len=queue_len, hindsight=hindsight)
		self.queue_len = queue_len
		self.batch_size = batch_size
		self.hindsight = hindsight
		self.gamma = gamma
		self.weight_end = weight_end
		self._step_fn = None
		self.input_dim = input_dim

	def update(self, pairs):
		preds = self.model(pairs[:,:self.input_dim])
		with torch.no_grad():
			target = self.compute_V_target(
				pairs[:,self.input_dim+2:-2],
				pairs[:,-2].unsqueeze(1),
				pairs[:,-1].unsqueeze(1),
			)

		loss =  100 *torch.mean(torch.square(preds - target))

		self.optimizer.zero_grad()
		loss.backward()
		# torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
		self.optimizer.step()
		return float(loss.item())

	def update_target(self):
		self.target_model.load_state_dict(self.model.state_dict())
		self.target_model.eval()

	def compute_V_target(self, next_state, l_s, g_s):
		# stop gradient
		with torch.no_grad():
			targets = self.gamma * torch.max(
				g_s, torch.min(l_s, self.target_model(next_state))
			) + (1.0 - self.gamma) * torch.max(l_s, g_s)
		return targets
	
if __name__ == 'main':
	scale_l = 0.5	
	pairs = np.load("data_generation/training_data_interrupted_different_n1.npy")
	pairs = np.load("quad_samples_pairs.npy")
	# pairs = pickle.load(open("quad_samples_pairs.pkl", "rb"))
	# for pair in pairs:
	# 	pair[:, -2] = np.tanh(pair[:, -2]/scale_l) 

	# # histogram of the last column of pairs (the violated column)
	# plt.figure(figsize=(6, 4))
	# plt.hist(pairs[:, -1], bins=20, edgecolor='black')
	# plt.title('Histogram of Violated Column in Pairs')
	# plt.xlabel('Violated Value')
	# plt.ylabel('Frequency')
	# plt.grid(axis='y', alpha=0.75)

	# # histogram of the second to last column of pairs (the reached column)
	# plt.figure(figsize=(6, 4))
	# plt.hist(pairs[:, -2], range=(-100, 0), bins=20, edgecolor='black')
	# plt.title('Histogram of Reached Column in Pairs')
	# plt.xlabel('Reached Value')
	# plt.ylabel('Frequency')
	# plt.grid(axis='y', alpha=0.75)	

	# plt.show()

	# print("Loaded samples:", sum([pair.shape[0] for pair in pairs]))
	print("Loaded pairs shape:", pairs.shape)

	print("Creating RALearner...")

	learner = RALearner(input_dim=13, queue_len=5000)
	# print("Adding transitions to buffer in batches of 200...")

	epochs = 5000
	shuffle_each_epoch = True

	batch_size = 4096
	n_samples = pairs.shape[0]
	n_batches = (n_samples + batch_size - 1) // batch_size

	query_grid, grid_shape = generate_query_grid(np.array([0,0,0,1,0,0,0,0,0,0,0,0,0]))

	# query_space = torch.Tensor(np.linspace(-2.2,2.2,100)).to(learner.device)

	# #search episode with succecsful transitions and print the target values for the first batch of pairs
	# for i, pair in enumerate(pairs):
	# 	successful_transitions = pair[pair[:,-2] < 0]
	# 	if successful_transitions.shape[0] > 0:
	# 		print("Example successful transitions found, at episode {}. Target values for the first batch:".format(i))
	# 		print(learner.compute_V_target(torch.Tensor(successful_transitions[:,13+2:-2]).to(learner.device), torch.Tensor(successful_transitions[:,-2].reshape(-1,1)).to(learner.device), torch.Tensor(successful_transitions[:,-1].reshape(-1,1)).to(learner.device)).cpu().numpy().flatten())
	# 		break

	# # plot trajectory of the first episode with successful transitions
	# plt.figure(figsize=(6, 6))
	# plt.plot(pairs[i][:,0], pairs[i][:,1], marker='o', markersize=3, label='Trajectory')
	# plt.title('Trajectory of First Episode with Successful Transitions')
	# plt.xlabel('x')
	# plt.ylabel('y')
	# plt.grid()
	# plt.legend()
	# plt.show()

	update_steps = 0
	for ep in tqdm(range(epochs)):
		if shuffle_each_epoch:
			perm = np.random.permutation(n_samples)
			epoch_pairs = pairs[perm]
		else:
			epoch_pairs = pairs
		epoch_loss = []
		# for batch_idx in range(n_batches):
		# 	start_idx = batch_idx * batch_size
		# 	end_idx = min(start_idx + batch_size, n_samples)
		# 	batch_indices = perm[start_idx:end_idx]

		# 	batch_states = torch.Tensor(pairs[batch_indices]).to(learner.device)
		# 	# print(f'Loss: {learner.update(batch_states):.6f}')
		# 	epoch_loss.append(learner.update(batch_states))
		# 	update_steps += 1
		# 	if update_steps % 20 == 0:
		# 		learner.update_target()
		# if ep % 5 == 0:
		# 	print(f"Epoch {ep+1}/{epochs}, Loss: {np.mean(epoch_loss):.6f}")
	
		for batch_idx in range(0, n_samples, batch_size):
			batch_states = torch.Tensor(epoch_pairs[batch_idx:batch_idx+batch_size]).to(learner.device)
			# print(f'Loss: {learner.update(batch_states):.6f}')
			epoch_loss.append(learner.update(batch_states))
			update_steps += 1
			if update_steps % 20 == 0:
				learner.update_target()
		if ep % 5 == 0:
			print(f"Epoch {ep+1}/{epochs}, Loss: {np.mean(epoch_loss):.6f}")
		
		# for i, pair in enumerate(pairs):
		# 	losses = []
		# 	batch_states = torch.Tensor(pair).to(learner.device)	
		# 	if i == 4:
		# 		print(pair[:,-2])
		# 		print(f"Target values: {learner.compute_V_target(batch_states[:,13+2:-2], batch_states[:,-2].unsqueeze(1), batch_states[:,-1].unsqueeze(1)).cpu().numpy().flatten()}") 
		# 	losses.append(learner.update(batch_states))
		# print(f"Epoch {ep+1}/{epochs}, Loss: {np.mean(losses):.6f}")
		# if ep % 10 == 0:
		# 	learner.update_target()

		if ep % 5 == 0:
			learner.model.eval()
			with torch.no_grad():
				grid_values = learner.model(torch.Tensor(query_grid).to(learner.device)).cpu().numpy().reshape(grid_shape)
				# grid_values = learner.model(query_space.unsqueeze(1)).cpu().numpy().flatten()
				# print('grid_values:', query_space.unsqueeze(1))
			learner.model.train()
			# plt.figure(figsize=(6, 6))
			# plt.plot(query_space.cpu().numpy(), grid_values, label='Learned V')
			# plt.grid(True)
			# plt.savefig(f"V_net.png")
			plot_V_XY(grid_values, log_learning=False)
			if shuffle_each_epoch:
				plt.savefig(f"V_net_quad_shuffle.png")
				torch.save(learner.model.state_dict(), "ra_value_function_shuffle.pth")
			else:
				plt.savefig(f"V_net_quad_no_shuffle.png")
				torch.save(learner.model.state_dict(), "ra_value_function_no_shuffle.pth")

			# with torch.no_grad():
			# 	query_state = torch.Tensor(np.array([[-4, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])).to(learner.device)
			# 	print(f'Value in center (0,0): {learner.model(query_state).cpu().numpy()}')
			
	# print(f"\nTraining complete. Final buffer size: {len(learner.buffer)}")
	print("Model saved to ra_value_function.pth")
