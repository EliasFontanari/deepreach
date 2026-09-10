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


def get_args():
    args_list = [
        "--mode", "test",
        "--experiment_class", "DeepReach",
        "--dynamics_class", "QuadcopterBox5",
        "--experiment_name", "QuadBox2",
        "--minWith", "target",
        "--pretrain",
        "--pretrain_iters", "1000",
        "--num_target_samples", "0",
        "--epochs_til_ckpt", "2000",
        "--set_mode", "avoid"
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

        p.add_argument('--device', type=str, default='cpu')
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

experiment_dir = os.path.join(opt.experiments_dir, opt.experiment_name)
if mode == 'test':
    # confirm that experiment dir already exists
    if not os.path.exists(experiment_dir):
        raise RuntimeError('Cannot run test mode: experiment directory not found!')

current_time = datetime.now()

# load original experiment settings
with open(os.path.join(experiment_dir, 'orig_opt.pickle'), 'rb') as opt_file:
    orig_opt = pickle.load(opt_file)


def print_orig_opt(orig_opt):
    """Stampa tutti gli attributi del run caricato da orig_opt.pickle.

    Sono le impostazioni con cui il modello e' stato *addestrato*: servono per sapere
    quale dinamica, quale tMax e quale deepReach_model stai effettivamente valutando.
    """
    attrs = vars(orig_opt)
    width = max(len(k) for k in attrs)
    name = getattr(orig_opt, 'experiment_name', '?')
    header = f" impostazioni del run '{name}' ({len(attrs)} attributi) "
    line_width = max(width + 40, len(header) + 8)
    print(header.center(line_width, '='))
    for k in sorted(attrs):
        print(f'  {k:<{width}} = {attrs[k]}')
    print('=' * line_width)


print_orig_opt(orig_opt)

# set the experiment seed
torch.manual_seed(orig_opt.seed)
random.seed(orig_opt.seed)
np.random.seed(orig_opt.seed)

dynamics_class = getattr(dynamics, orig_opt.dynamics_class)
dynamics_params = [argname for argname in inspect.signature(dynamics_class).parameters.keys() if argname != 'self']
# i run vecchi sono stati picklati prima che 'device' entrasse nella firma delle dinamiche:
# prendo dall'orig_opt solo i parametri che ci sono davvero
dynamics_kwargs = {argname: getattr(orig_opt, argname) for argname in dynamics_params if hasattr(orig_opt, argname)}
# qui si valuta soltanto, e il modello sta su cpu: tengo su cpu anche i tensori della dinamica
if 'device' in dynamics_params:
    dynamics_kwargs['device'] = 'cpu'
dynamics_obj = dynamics_class(**dynamics_kwargs)
dynamics_obj.deepReach_model = orig_opt.deepReach_model

model = modules.SingleBVPNet(in_features=dynamics_obj.input_dim, out_features=1, type=orig_opt.model, mode=orig_opt.model_mode,
                             final_layer_factor=1., hidden_features=orig_opt.num_nl, num_hidden_layers=orig_opt.num_hl, periodic_transform_fn=dynamics_obj.periodic_transform_fn)

model.to('cpu')

# checkpoint = experiment_dir + '/training/checkpoints/model_epoch_104000.pth'
checkpoint = experiment_dir + '/training/checkpoints/model_final.pth'

model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])


def _move_to_cpu(obj):
    for name, val in vars(obj).items():
        if isinstance(val, torch.Tensor):
            setattr(obj, name, val.cpu())
        elif isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, torch.Tensor):
                    val[k] = v.cpu()
        elif isinstance(val, (list, tuple)):
            new_seq = [v.cpu() if isinstance(v, torch.Tensor) else v for v in val]
            setattr(obj, name, type(val)(new_seq))


_move_to_cpu(dynamics_obj)

import matplotlib.ticker as ticker


def plot_state(model, device, dynamics, dataslice: np.ndarray, time: float, x_axis: int, y_axis: int, x_resolution: int, y_resolution: int, n_ticks: int = 15):
    model.eval()
    model.requires_grad_(False)

    plot_config = dynamics.plot_config()

    state_test_range = dynamics.state_range_
    x_min, x_max = state_test_range[x_axis]
    y_min, y_max = state_test_range[y_axis]

    xs = torch.linspace(x_min, x_max, x_resolution)
    ys = torch.linspace(y_min, y_max, y_resolution)
    xys = torch.cartesian_prod(xs, ys)

    fig = plt.figure()
    ax = fig.gca()
    coords = torch.zeros(x_resolution*y_resolution, dynamics.state_dim + 1)
    coords[:, 0] = time
    coords[:, 1:] = torch.tensor(dataslice)
    coords[:, 1 + x_axis] = xys[:, 0]
    coords[:, 1 + y_axis] = xys[:, 1]
    coords[:, 1:] = dynamics.equivalent_wrapped_state(coords[:, 1:])

    with torch.no_grad():
        model_results = model({'coords': dynamics.coord_to_input(coords.to('cpu')).to('cpu')})
        values = dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())

    plt.title(f't = {time}, state = {dataslice}')
    plt.xlabel(f"{plot_config['state_labels'][x_axis]}")
    plt.ylabel(f"{plot_config['state_labels'][y_axis]}")

    s = plt.imshow(1*(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T), cmap='inferno', origin='lower', extent=(x_min.cpu(), x_max.cpu(), y_min.cpu(), y_max.cpu()))
    fig.colorbar(s)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(n_ticks))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(n_ticks))
    plt.grid(True)

    fig2 = plt.figure()
    ax2 = fig2.gca()

    plt.title(f't = {time}, state = {dataslice}')
    plt.xlabel(f"{plot_config['state_labels'][x_axis]}")
    plt.ylabel(f"{plot_config['state_labels'][y_axis]}")

    s = plt.imshow(1*(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T <= 0), cmap='bwr', origin='lower', extent=(x_min.cpu(), x_max.cpu(), y_min.cpu(), y_max.cpu()))
    fig2.colorbar(s)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(n_ticks))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(n_ticks))
    plt.grid(True)

    return values.detach().cpu().numpy(), fig, fig2


# lo stato del QuadcopterBox2 e' cosi' composto (13 componenti):
# [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, w_x, w_y, w_z, box]
# dove box e' la semi-larghezza del cubo in cui il quadricottero deve restare
# (il cubo e' [-box, box] su x, y e z), con box in [0.15, 4.0].
slice_state = np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 2.])
tMax = orig_opt.tMax  # 0.6 per questo run

val2, fig_value, fig_mask = plot_state(model, 'cpu', dynamics_obj, slice_state, tMax, 0, 1, 100, 100, n_ticks=15)

# savefig() salva la figura *corrente*, che e' l'ultima creata: salvo esplicitamente entrambe
fig_value.savefig('test_quadcopter_value.png')
fig_mask.savefig('test_quadcopter_mask.png')
plt.show()


# per valutare la value function V(t, x) per un singolo stato, usa la funzione evaluate_V.
# Come tempo passa sempre quello massimo, 0.6 in questo caso.
# se V ritornata e' negativa, lo stato e' insicuro, se positiva, lo stato e' sicuro.

def evaluate_V(model, dynamics, state, time):
    """
    Valuta la value function V(t, x) per un singolo stato e istante temporale.

    state: array-like di lunghezza dynamics.state_dim
    time: float, istante temporale
    """
    model.eval()
    model.requires_grad_(False)

    state = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
    # psi e' periodico: lo riporto in [-pi, pi] come fa il resto della pipeline,
    # altrimenti la rete viene interrogata fuori dal suo range di training
    state = dynamics.equivalent_wrapped_state(state)
    coords = torch.zeros(1, dynamics.state_dim + 1)
    coords[:, 0] = time
    coords[:, 1:] = state

    with torch.no_grad():
        model_results = model({'coords': dynamics.coord_to_input(coords.to('cpu')).to('cpu')})
        value = dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())

    return value.item()


def evaluate_V_batch(model, dynamics, states, time):
    """
    Come evaluate_V, ma per un insieme di stati.

    states: array-like di forma (N, dynamics.state_dim)
    time: float, istante temporale
    ritorna un np.ndarray di forma (N,) con i valori di V
    """
    model.eval()
    model.requires_grad_(False)

    states = torch.as_tensor(states, dtype=torch.float32).reshape(-1, dynamics.state_dim)
    states = dynamics.equivalent_wrapped_state(states)
    coords = torch.zeros(states.shape[0], dynamics.state_dim + 1)
    coords[:, 0] = time
    coords[:, 1:] = states

    with torch.no_grad():
        model_results = model({'coords': dynamics.coord_to_input(coords.to('cpu')).to('cpu')})
        values = dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())

    return values.detach().cpu().numpy()


v = evaluate_V(model, dynamics_obj, [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 2.], tMax)
print(v)
