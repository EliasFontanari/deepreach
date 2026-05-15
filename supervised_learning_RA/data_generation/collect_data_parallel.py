import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from system_dyn import x_next,pi,phi,c,r,gen_noise,u_max,d_max,x_th,gamma,distr,x_min,x_max,state_size,failure_max,target_max
import multiprocessing

episodes = 1_000_000
length = 200

print(f'State size : {state_size}')

cores = 20

manager = multiprocessing.Manager()
succ = manager.Value('i', 0)
fails =  manager.Value('i', 0)
executions= manager.Value('i', 0)

counter_lock = multiprocessing.Lock()

def generate_pairs(data):
    """
    Execute on horizon x (states + reached + violated) data
    """
    pairs = []
    for j in range(data.shape[1] - 1):
        if (data[j, :] != np.zeros(data.shape[1])).any() and (
            data[j + 1, :] != np.zeros(data.shape[1])
        ).any():
            pairs.append(np.hstack((data[ j, :], data[j + 1, :])))
    return np.array(pairs)

def simulate_traj(dummy):
    with counter_lock:
        executions.value +=1
        if executions.value % 1000 == 0:
            print(f'Executions {executions.value}/{episodes}')
    data = np.zeros((length,state_size+2))
    x = np.random.uniform(x_min,x_max,state_size)
    # x = np.array([np.random.uniform(x_min,x_max)]*state_size)

    data[0] = np.hstack((x,[((r(x) > 0).all() and (c(x) >= 0).all()), (c(x) < 0).any()]))
    if ((r(x) > 0).all() and (c(x) >= 0).all()):
        data[1] = np.hstack((x,[True, False]))
        # print('succ')
        with counter_lock:
            succ.value +=1
        
        return generate_pairs(data)
    if not(data[0][-2]):
        for j in range(length-1):
            x = x_next(x,pi(x),gen_noise())

            data[j+1] = np.hstack((x, [((r(x) > 0).all() and (c(x) >= 0).all()), (c(x) < 0).any()]))
            if ((r(x) > 0).all() and (c(x) >= 0).all()):
                with counter_lock:
                    succ.value +=1
                return generate_pairs(data)
            if (c(x) < 0).any():
                # print(f'x < -2   {x}')
                # fails += 1
                with counter_lock:
                    fails.value +=1
                return generate_pairs(data)
            if (x > x_max).any() or (x < x_min).any():
                data[j+1] = np.hstack((x, [False,True]))
                with counter_lock:
                    fails.value +=1
                return generate_pairs(data)      
        return generate_pairs(data)    

with multiprocessing.Pool(processes=cores) as pool:
        # `range(num_tasks)` generates a sequence of 100 dummy values (0-99)
        res = pool.map(simulate_traj, range(episodes))


print("All tasks are completed.")

fails = fails.value
succ = succ.value

res = np.vstack((res))

# counter_succ = 0
# ratio_constant = 3
# data_tmp = np.zeros((ratio_constant*fails,length,state_size+2))

# while counter_succ < ratio_constant *fails:
#     # print(f'Counter {counter}')
#     if succ % 1000 == 0:
#         print(f'Succ {succ}, fails {fails}, ratio {succ/fails}')
#     x = np.random.uniform(failure_max,target_max,state_size)

#     data_tmp[counter_succ,0] = np.hstack((x,[((r(x) > 0).all() and (c(x) >= 0).all()), (c(x) < 0).any()]))
#     # if ((r(x) > 0).all() and (c(x) >= 0).all()):
#     #     data_tmp[counter_succ,1] = np.hstack((x,[True, False]))
#     #     succ+=1
#     #     counter_succ += 1

#     #     continue
    
#     reached_length = 0
#     reached_old = 0
    
#     for j in range(1):
#         x_dyn = x_next(x,pi(x),gen_noise())
#         reached_indx = np.where((r(x_dyn) > 0) & (c(x_dyn) >= 0))[0] 
#         reached_length = reached_indx.shape[0]
#         x = x_dyn
#         if reached_length > 0:
#             if reached_length > reached_old:              
#                 x_reached = np.copy(x)
#                 x[reached_indx] = x_reached[reached_indx]
#                 reached_old = reached_length
#             else:
#                 x[reached_indx] = data_tmp[counter_succ,j,reached_indx]
        
#         # print(f'Reached {reached_indx}')
#         if ((r(x) > 0).all() and (c(x) >= 0).all()):
#             succ += 1
#             data_tmp[counter_succ,j+1] = np.hstack((x, [((r(x) > 0).all() and (c(x) >= 0).all()), (c(x) < 0).any()]))
#             counter_succ += 1
#             # print('SUCC')
#             break
#         if (c(x) < 0).any():
#             # print(f'x < -2   {x} fails')
#             # fails += 1
#             break
#         if (x > x_max).any() or (x < x_min).any():
#             # data_tmp[0,j+1] = np.hstack((x, [False,True]))
#             # fails += 1
#             break
        
# res = np.vstack((res,data_tmp))

print(f'Successes {succ} , failures {fails}, shape {res.shape}')

np.save(f'data_pairs_state_size{state_size}.npy',res)
