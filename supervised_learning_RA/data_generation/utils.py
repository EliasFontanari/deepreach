import numpy as np
# import jax
# import jax.numpy as jnp

import shutil

def generate_pairs(data):
    """
    Execute on n_episodes x horizon x (states + reached + violated) data
    """
    pairs = []
    for i in range(0,data.shape[0],2):
        for j in range(data.shape[1] - 2):
            if (data[i, j, :] != np.zeros(data.shape[2])).any() and (
                data[i, j + 1, :] != np.zeros(data.shape[2])
            ).any():
                pairs.append(np.hstack((data[i, j, :], data[i, j + 1, :])))
    return np.array(pairs)

# def augment_dataset(data_pairs, region_start, region_end, number_artificial_samples):
#     state_size = int(data_pairs.shape[1]/2 -2)
#     samples_list = []
#     for i in range(data_pairs.shape[0]):
#         if (region_start <= data_pairs[i,:state_size]).all() and (region_end >= data_pairs[i,:state_size]).all():
#             samples_list.append(data_pairs[i])
#     counter = 0
#     artificial_array = np.zeros((number_artificial_samples,data_pairs.shape[1]))
#     while counter < number_artificial_samples:
#         random_indx = np.random.randint(0,len(samples_list))
#         artificial_array[counter] = samples_list[random_indx]
#         counter += 1
#     return np.vstack((data_pairs,artificial_array))

# # ====================================
# #         State Normalization
# # ====================================
# def normalize_states(data_pairs):
#     state_size = int(data_pairs.shape[1]/2 -2)
#     mean_x = np.mean(data_pairs[:,:state_size], axis=0)
#     std_x = np.std(data_pairs[:,:state_size], axis=0)

#     mean_x_next = np.mean(data_pairs[:,-(state_size+2):-2], axis=0)
#     std_x_next = np.std(data_pairs[:,-(state_size+2):-2], axis=0) 

#     mean_tot = (mean_x + mean_x_next)/2
#     std_tot = (std_x + std_x_next)/2

#     data_pairs[:,:state_size] = (data_pairs[:,:state_size] - mean_tot) / (std_tot + 1e-8)
#     data_pairs[:,-(state_size+2):-2] = (data_pairs[:,-(state_size+2):-2] - mean_tot) / (std_tot + 1e-8)
#     return mean_tot,std_tot

# # @jax.jit
# def get_batches(pairs, batch_size, full_batches_num, partial_batch_length, key):
#     # Get the dataset length
#     dataset_len = pairs.shape[0]

#     # Generate a permutation of indices
#     rand_indx = jax.random.permutation(key, dataset_len)

#     # Shuffle the dataset using the permuted indices
#     shuffled_pairs = pairs[rand_indx]

#     # Split the shuffled data into full batches
#     full_batches = jnp.array_split(shuffled_pairs[:full_batches_num * batch_size], full_batches_num)

    
#     last_batch = shuffled_pairs[-partial_batch_length:]
#     full_batches.append(last_batch)

#     return full_batches

# def safe_np_save(filepath, array, min_free_gb=1.0):
#     free_bytes = shutil.disk_usage(filepath.rsplit('/', 1)[0] or '.').free
#     array_bytes = array.nbytes
#     min_free_bytes = min_free_gb * 1024**3

#     if array_bytes + min_free_bytes > free_bytes:
#         raise RuntimeError(
#             f"Not enough disk space: need {array_bytes/1e9:.2f} GB, "
#             f"only {free_bytes/1e9:.2f} GB free"
#         )
#     np.save(filepath, array)