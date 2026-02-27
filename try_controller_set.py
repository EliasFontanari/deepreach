import configargparse
import argparse
import inspect
import os
import torch
import shutil
import random
import numpy as np
import pickle

from datetime import datetime
from dynamics import dynamics 
from experiments import experiments
from utils import modules, dataio, losses

import matplotlib.pyplot as plt

# import sys
# sys.argv = [
#    '--mode', 'train',
#    '--experiment_class', 'DeepReach',
#    '--experiment_name', 'QUADRA',
#    '--dynamics_class','QuadrotorReachAvoid'

# ]

def get_args():
    args_list = [
        "--mode", "test",
        "--experiment_class", "DeepReach",
        "--dynamics_class", "QuadrotorReachAvoid",
        "--experiment_name", "QUADRA6",
        "--minWith", "target",
        "--pretrain",
        "--pretrain_iters", "10000",
        "--num_target_samples", "5000",
        "--epochs_til_ckpt", "200",
    ]

    p = configargparse.ArgumentParser()
    p.add_argument('-c', '--config_filepath', required=False, is_config_file=True)
    p.add_argument('--mode', type=str, required=True, choices=['all', 'train', 'test'])
    p.add_argument('--experiments_dir', type=str, default='./runs')
    p.add_argument('--experiment_name', type=str, required=True)
    p.add_argument('--use_wandb', default=False, action='store_true')

    # Use the SAME args_list to detect mode
    mode = p.parse_known_args(args_list)[0].mode

    if (mode == 'all') or (mode == 'train'):
        p.add_argument('--seed', type=int, default=0)
        experiment_classes_dict = {name: clss for name, clss in inspect.getmembers(experiments, inspect.isclass) if clss.__bases__[0] == experiments.Experiment}
        p.add_argument('--experiment_class', type=str, default='DeepReach', choices=experiment_classes_dict.keys())
        experiment_class = experiment_classes_dict[p.parse_known_args(args_list)[0].experiment_class]
        experiment_params = {name: param for name, param in inspect.signature(experiment_class.init_special).parameters.items() if name != 'self'}
        for param in experiment_params.keys():
            p.add_argument('--' + param, type=experiment_params[param].annotation, required=True)

        p.add_argument('--device', type=str, default='cuda:0')
        p.add_argument('--numpoints', type=int, default=65000)
        p.add_argument('--pretrain', action='store_true', default=False)
        p.add_argument('--pretrain_iters', type=int, default=2000)
        p.add_argument('--tMin', type=float, default=0.0)
        p.add_argument('--tMax', type=float, default=1.0)
        p.add_argument('--counter_start', type=int, default=0)
        p.add_argument('--counter_end', type=int, default=-1)
        p.add_argument('--num_src_samples', type=int, default=1000)
        p.add_argument('--num_target_samples', type=int, default=0)
        p.add_argument('--model', type=str, default='sine', choices=['sine', 'tanh', 'sigmoid', 'relu'])
        p.add_argument('--model_mode', type=str, default='mlp', choices=['mlp', 'rbf', 'pinn'])
        p.add_argument('--num_hl', type=int, default=3)
        p.add_argument('--num_nl', type=int, default=512)
        p.add_argument('--deepreach_model', type=str, default='exact', choices=['exact', 'diff', 'vanilla'])
        p.add_argument('--epochs_til_ckpt', type=int, default=1000)
        p.add_argument('--steps_til_summary', type=int, default=100)
        p.add_argument('--batch_size', type=int, default=1)
        p.add_argument('--lr', type=float, default=2e-5)
        p.add_argument('--num_epochs', type=int, default=100000)
        p.add_argument('--clip_grad', default=0.0, type=float)
        p.add_argument('--use_lbfgs', default=False, type=bool)
        p.add_argument('--adj_rel_grads', default=True, type=bool)
        p.add_argument('--dirichlet_loss_divisor', default=1.0, type=float)
        p.add_argument('--use_CSL', default=False, action='store_true')
        p.add_argument('--CSL_lr', type=float, default=2e-5)
        p.add_argument('--CSL_dt', type=float, default=0.0025)
        p.add_argument('--epochs_til_CSL', type=int, default=10000)
        p.add_argument('--num_CSL_samples', type=int, default=1000000)
        p.add_argument('--CSL_loss_frac_cutoff', type=float, default=0.1)
        p.add_argument('--max_CSL_epochs', type=int, default=100)
        p.add_argument('--CSL_loss_weight', type=float, default=1.0)
        p.add_argument('--CSL_batch_size', type=int, default=1000)
        p.add_argument('--val_x_resolution', type=int, default=200)
        p.add_argument('--val_y_resolution', type=int, default=200)
        p.add_argument('--val_z_resolution', type=int, default=5)
        p.add_argument('--val_time_resolution', type=int, default=3)
        p.add_argument('--minWith', type=str, required=True, choices=['none', 'zero', 'target'])

        dynamics_classes_dict = {name: clss for name, clss in inspect.getmembers(dynamics, inspect.isclass) if clss.__bases__[0] == dynamics.Dynamics}
        p.add_argument('--dynamics_class', type=str, required=True, choices=dynamics_classes_dict.keys())
        dynamics_class = dynamics_classes_dict[p.parse_known_args(args_list)[0].dynamics_class]
        dynamics_params = {name: param for name, param in inspect.signature(dynamics_class).parameters.items() if name != 'self'}
        for param in dynamics_params.keys():
            if dynamics_params[param].annotation is bool:
                p.add_argument('--' + param, type=dynamics_params[param].annotation, default=False)
            else:
                p.add_argument('--' + param, type=dynamics_params[param].annotation, required=True)

    if (mode == 'all') or (mode == 'test'):
        p.add_argument('--dt', type=float, default=0.0025)
        p.add_argument('--checkpoint_toload', type=int, default=None)
        p.add_argument('--num_scenarios', type=int, default=100000)
        p.add_argument('--num_violations', type=int, default=1000)
        p.add_argument('--control_type', type=str, default='value', choices=['value', 'ttr', 'init_ttr'])
        p.add_argument('--data_step', type=str, default='run_basic_recovery', choices=['plot_violations', 'run_basic_recovery', 'plot_basic_recovery', 'collect_samples', 'train_binner', 'run_binned_recovery', 'plot_binned_recovery', 'plot_cost_function'])

    # Parse and RETURN the namespace, not the parser
    opt, _ = p.parse_known_args(args_list)
    return opt

opt = get_args()
mode = opt.mode 

# %%
experiment_dir = os.path.join(opt.experiments_dir, opt.experiment_name)
if mode == 'test':
    # confirm that experiment dir already exists
    if not os.path.exists(experiment_dir):
        raise RuntimeError('Cannot run test mode: experiment directory not found!')

current_time = datetime.now()

# load original experiment settings
with open(os.path.join(experiment_dir, 'orig_opt.pickle'), 'rb') as opt_file:
    orig_opt = pickle.load(opt_file)

# set the experiment seed
torch.manual_seed(orig_opt.seed)
random.seed(orig_opt.seed)
np.random.seed(orig_opt.seed)


# %%
dynamics_class = getattr(dynamics, orig_opt.dynamics_class)
dynamics_obj = dynamics_class(**{argname: getattr(orig_opt, argname) for argname in inspect.signature(dynamics_class).parameters.keys() if argname != 'self'})
dynamics_obj.deepreach_model=orig_opt.deepreach_model

model = modules.SingleBVPNet(in_features=dynamics_obj.input_dim, out_features=1, type=orig_opt.model, mode=orig_opt.model_mode,
                             final_layer_factor=1., hidden_features=orig_opt.num_nl, num_hidden_layers=orig_opt.num_hl)
model.to(orig_opt.device)

checkpoint = experiment_dir + '/training/checkpoints/model_epoch_300000.pth'

model.load_state_dict(torch.load(checkpoint)['model'])

# %%
model.training

# %%
def plot_state(model, device, dynamics, dataslice: np.ndarray, time: float, x_axis: int, y_axis: int, x_resolution: int, y_resolution:int):
        model.eval()
        model.requires_grad_(False)

        plot_config = dynamics.plot_config()

        state_test_range = dynamics.state_range_
        print(state_test_range[0])
        x_min, x_max = state_test_range[x_axis]
        y_min, y_max = state_test_range[y_axis]

        print(f'xmin xmax {x_min}, {x_max}')
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        xys = torch.cartesian_prod(xs, ys)
        
        fig = plt.figure() 
        coords = torch.zeros(x_resolution*y_resolution, dynamics.state_dim + 1)
        coords[:, 0] = time
        coords[:, 1:] = torch.tensor(dataslice)
        coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
        coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]

        with torch.no_grad():
            model_results = model({'coords': dynamics.coord_to_input(coords.to(device))})
            values = dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
        
        plt.title(f't = {time}, state = {dataslice}')
        plt.xlabel(f"{plot_config['state_labels'][plot_config['x_axis_idx']]}")
        plt.ylabel(f"{plot_config['state_labels'][plot_config['y_axis_idx']]}")


        s = plt.imshow(1*(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T), cmap='inferno', origin='lower', extent=(x_min.cpu(), x_max.cpu(), y_min.cpu(), y_max.cpu()))
        # s = ax.imshow(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
        fig.colorbar(s) 

        plt.figure()
        coords = torch.zeros(x_resolution*y_resolution, dynamics.state_dim + 1)
        coords[:, 0] = time
        coords[:, 1:] = torch.tensor(dataslice)
        coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
        coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
        
        plt.title(f't = {time}, state = {dataslice}')
        plt.xlabel(f"{plot_config['state_labels'][plot_config['x_axis_idx']]}")
        plt.ylabel(f"{plot_config['state_labels'][plot_config['y_axis_idx']]}")


        s = plt.imshow(1*(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T <= 0), cmap='bwr', origin='lower', extent=(x_min.cpu(), x_max.cpu(), y_min.cpu(), y_max.cpu()))
        # s = ax.imshow(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
        fig.colorbar(s) 

        plt.show()
        return values.detach().cpu().numpy()

val = plot_state(model,orig_opt.device,dynamics_obj,np.array([1.,0,0,1,0.0,0.0,0.,1,0,0,0,0,0]),0.05,7,8,100,100)

# verification

# sample a state in reach-avoid set
value = 10
while value > 0:
    # Split min and max
    low = dynamics_obj.state_range_[:, 0]
    high = dynamics_obj.state_range_[:, 1]

    # Uniform sample within bounds
    state = low + (high - low) * torch.rand_like(low)
    state = torch.tensor([1.,0,0,1,0.0,0.0,0.,1.8,1.4,0,.0,0,0])
    state = dynamics_obj.equivalent_wrapped_state(state)
    time = 0.4
    coord = torch.zeros(1, dynamics_obj.state_dim + 1)
    coord[0,0] = time
    coord[0,1:] = state
    with torch.no_grad():
        model_result = model({'coords': dynamics_obj.coord_to_input(coord.to(orig_opt.device))})
        value = dynamics_obj.io_to_value(model_result['model_in'].detach(), model_result['model_out'].squeeze(dim=-1).detach())
    if bool(value < 0): print(f'State {state} value {value}')

# simulation 
dt = 0.001
t_max = 0.4
n_step = int(t_max/dt)
traj = torch.zeros(n_step+1,dynamics_obj.state_dim)
traj_u = torch.zeros(n_step,dynamics_obj.control_dim)
traj[0] = state
device = orig_opt.device
time = 0
for i in range(n_step):
    coord = torch.zeros(1,dynamics_obj.state_dim+1)
    coord[0] = time
    coord[0,1:] = traj[i]
    traj_coord = model({'coords': dynamics_obj.coord_to_input(coord.to(device))})
    dvds = dynamics_obj.io_to_dv(traj_coord['model_in'], traj_coord['model_out'].squeeze(dim=-1)).detach()
    ctrl = dynamics_obj.optimal_control(coord[:, 1:].to(device), dvds[..., 1:].to(device))
    traj_u[i] = ctrl
    traj[i+1] = dynamics_obj.rk4_step_quad(traj[i],ctrl,0,dt)

    reach_val = dynamics_obj.reach_fn(traj[i+1])
    if bool(reach_val < 0):
        print(f'Reached at time step {i} state{traj[i+1]}')

    time += dt

traj_x = traj.detach().cpu().numpy()
traj_u = traj_u.detach().cpu().numpy()

import matplotlib.pyplot as plt

labels_x = ['x','y','z','q_w','q_x','q_y','q_z','v_x','v_y','v_z','w_x','w_y','w_z']
labels_u = ['f_tot','alpha_x','alpha_y','alpha_z']
def plot_x_trajectory(traj):
    """
    traj: (n, 13) tensor or numpy array
    """

    n, dim = traj.shape
    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    axes = axes.flatten()

    time = np.arange(n)*dt

    for i in range(dim):
        axes[i].plot(time, traj[:, i])
        axes[i].set_title(f"{labels_x[i]}")
        axes[i].set_xlabel("Time")
        axes[i].grid(True)

    # Turn off unused subplots (last 3)
    for i in range(13, 16):
        axes[i].axis("off")

    plt.tight_layout()

def plot_u_trajectory(traj):
    """
    traj: (n-1, 4) tensor or numpy array
    """

    n, dim = traj.shape
    fig, axes = plt.subplots(1, 4, figsize=(14, 10))
    axes = axes.flatten()

    time = np.arange(n)*dt

    for i in range(dim):
        axes[i].plot(time, traj[:, i])
        axes[i].set_title(f"{labels_u[i]}")
        axes[i].set_xlabel("Time")
        axes[i].grid(True)

    plt.tight_layout()

plot_x_trajectory(traj_x)
plot_u_trajectory(traj_u)
plt.show()

    
