import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
from system_dyn import x_next,pi,phi,c,r,gen_noise,u_max,d_max,x_th,gamma,distr,x_min,x_max,state_size,failure_max,target_max

episodes = 30_000
length = 200

data = np.zeros((episodes,length,state_size+2))
succ=0
fails=0

print(f'State size : {state_size}')