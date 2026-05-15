import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from system_dyn import x_next,pi,phi,c,r,gen_noise,u_max,d_max,x_th,gamma,distr,x_min,x_max,state_size,grid_max,grid_min

grid_state_size = 1000
state_grid = np.linspace(grid_min,grid_max,grid_state_size+1)

V_prob = np.load('data/V_prob_x_th:-1.5_u_max:_1_d_max:_4_distr_gaussian.npy')
state_grid = np.load('data/grid.npy')

def step_tabular(x):
    if isinstance(x, np.ndarray):
        x_n = np.copy(x)
        u = pi(x)
        noises = gen_noise()
        for i in range(x.shape[0]):
            x_n[i] = x_next(x[i],u[i],noises[i])
            index_x_n = np.argmin(np.abs(state_grid - x_n[i]))
            x_n[i] = state_grid[index_x_n]
        return x_next(x,pi(x),gen_noise())

def get_value(x):
    V_x = 1
    for i in range(x.shape[0]):
        indice_x = np.argmin(np.abs(state_grid - x[i]))
        V_x *= V_prob[indice_x]
    return V_x

# def get_value(x):
#     # Find the indices of the nearest values in state_grid for each element in x
#     indices = np.abs(state_grid[:, None] - x).argmin(axis=0)
    
#     # Use the indices to access the corresponding values in V_prob
#     V_x = np.prod(V_prob[indices])
    
#     return V_x

episodes = 30_000
length = 200

data = np.zeros((episodes,length,state_size+1))
succ=0
fails=0

print(f'State size : {state_size}')

for i in tqdm(range(int(episodes))):
    x = np.random.uniform(x_min,x_max,state_size)
    if state_size == 1:
        x = x.reshape(1)
    data[i,0] = np.hstack((x,get_value(x)))

    reached_length = 0
    reached_old = 0

    if ((r(x) > 0).all() and (c(x) >= 0).all()):
        succ+=1
        continue
    
    # if (c(x) >= 0).all():
    for j in range(length-1):
        x_dyn = step_tabular(x)
        reached_indx = np.where((r(x_dyn) > 0) & (c(x_dyn) >= 0))[0] 
        reached_length = reached_indx.shape[0]
        x = x_dyn
        if reached_length > 0:
            if reached_length > reached_old:              
                x_reached = np.copy(x)
                x[reached_indx] = x_reached[reached_indx]
                reached_old = reached_length
            else:
                x[reached_indx] = data[i,j,reached_indx]
        
        # print(f'Reached {reached_indx}')
        data[i,j+1] = np.hstack((x,get_value(x)))
        if ((r(x) > 0).all() and (c(x) >= 0).all()):
            succ += 1
            # print('SUCC')
            break
        if (c(x) < 0).any():
            fails += 1
            break
        if (x > x_max).any() or (x < x_min).any():
            data[i,j+1] = np.hstack((x, 0))
            fails += 1
            break
    # else: 
    #     # for j in range(length-1):
    #     #     x_dyn = step_tabular(x)
    #     #     reached_indx = np.where((r(x_dyn) > 0) & (c(x_dyn) >= 0))[0] 
    #     #     reached_length = reached_indx.shape[0]
    #     #     x = x_dyn
    #     #     if reached_length > 0:
    #     #         if reached_length > reached_old:              
    #     #             x_reached = np.copy(x)
    #     #             x[reached_indx] = x_reached[reached_indx]
    #     #             reached_old = reached_length
    #     #         else:
    #     #             x[reached_indx] = data[i,j,reached_indx]
            
    #         # print(f'Reached {reached_indx}')
    #         # data[i,j+1] = np.hstack((x,get_value(x)))
    #     fails += 1
    #     # break

# for i in tqdm(range(int(episodes/2),episodes)):
#     x = np.random.uniform(x_min,-2,state_size)
#     if state_size == 1:
#         x = x.reshape(1)
#     data[i,0] = np.hstack((x,get_value(x)))

#     reached_length = 0
#     reached_old = 0
    
#     # if (c(x) >= 0).all():
#     for j in range(length-1):
#         x_dyn = step_tabular(x)
#         reached_indx = np.where((r(x_dyn) > 0) & (c(x_dyn) >= 0))[0] 
#         reached_length = reached_indx.shape[0]
#         x = x_dyn
#         if reached_length > 0:
#             if reached_length > reached_old:              
#                 x_reached = np.copy(x)
#                 x[reached_indx] = x_reached[reached_indx]
#                 reached_old = reached_length
#             else:
#                 x[reached_indx] = data[i,j,reached_indx]
        
#         # print(f'Reached {reached_indx}')
#         data[i,j+1] = np.hstack((x,get_value(x)))
        


print(f'Successes {succ} , failures {fails}')
np.save(f'data/training_data_interrupted__tabular_n{state_size}.npy',data)



