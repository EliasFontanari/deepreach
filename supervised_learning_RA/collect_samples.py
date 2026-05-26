import os
import concurrent.futures
import multiprocessing as mp

import numpy as np
from quad_ocp import QuadOCP
import matplotlib.pyplot as plt
import tqdm as tqdm
import pickle
n_episodes = 80000
MPC_frequency = 10
T_tot = 10
n_workers = 4

def generate_pairs(data):
    """
    Execute on n_episodes x horizon x (states + reached + violated) data
    """
    pairs = []
    for j in range(0, data.shape[0] - 1, 1):
        if (data[j, :] != np.zeros(data.shape[1])).any() and (
            data[j + 1, :] != np.zeros(data.shape[1])
        ).any():
            pairs.append(np.hstack((data[j, :], data[j + 1, :])))
    return np.array(pairs)

def rollout_episode(episode_idx, seed):
    rng = np.random.default_rng(seed)
    solver = QuadOCP()
    step_fn = solver.rk4_step_fn()
    rollout_states = []

    x0 = rng.uniform(low=solver.state_range[:, 0], high=solver.state_range[:, 1])
    x0[3:7] /= np.linalg.norm(x0[3:7])

    rollout_states.append(np.hstack((x0, solver.compute_l(x0), solver.compute_g(x0))))

    sol_u = None
    
    success = False
    for i in range(int(T_tot / solver.dt)):
        if i % MPC_frequency == 0:
            try:
                sol_u = solver.solve(rollout_states[-1][:x0.shape[0]])["traj_u"][:, :MPC_frequency]
            except:
                print(f"Episode {episode_idx}: MPC failed at step {i}, state: {rollout_states[-1]}")
                return {
                "pairs":generate_pairs(np.array(rollout_states)),
                "success": False,
                "episode_length": i,
            }
        x_next = np.array(step_fn(rollout_states[-1][:x0.shape[0]], sol_u[:, i % MPC_frequency])).squeeze()
        l_s = solver.compute_l(x_next)
        g_s = solver.compute_g(x_next)
        rollout_states.append(np.hstack((x_next, l_s, g_s)))

        if g_s >= 0.0:
            for j in range(min(10, i)):
                rollout_states[-j][-1] = 1 - 0.2 * j
            print(f"Episode {episode_idx}: Violation at step {i}, state: {rollout_states[-1]}")
            return {
                "pairs": generate_pairs(np.array(rollout_states)),
                "success": False,
                "episode_length": i,
            }
        if rollout_states[-1][-2] < 0 and not success:
            print(f"Episode {episode_idx}: Success at step {i}, state: {rollout_states[-1]}")
            success = True
            return {
            "pairs": generate_pairs(np.array(rollout_states)),
            "success": success,
            "episode_length": i,
        }

    rollout_states[-1][-2] = 1
    return {
        "pairs": generate_pairs(np.array(rollout_states)),
        "success": success,
        "episode_length": i,
    }


def main():
    seed_base = 12345
    ctx = mp.get_context("spawn")
    futures = []
    results = []
    successes = 0
    failures = 0

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx
    ) as executor:
        for episode in range(n_episodes):
            futures.append(executor.submit(rollout_episode, episode, seed_base + episode))

        for fut in tqdm.tqdm(concurrent.futures.as_completed(futures), total=n_episodes):
            res = fut.result()
            results.append(res)
            if res["success"]:
                successes += 1
            else:
                failures += 1

    print(f'Lengths of results: {len(results)}')
    x_traj = np.concatenate([r["pairs"] for r in results if r["pairs"].shape[0] > 0], axis=0)
    # x_traj = [r['pairs'] for r in results if r['pairs'].shape[0] > 0]

    
    # pickle.dump(x_traj, open("quad_samples_pairs.pkl", "wb"))
    
    episodes_lengths = [r["episode_length"] for r in results]
    
    np.save("quad_samples_pairs.npy", x_traj)
    
    print(f"Average episode length: {np.mean(episodes_lengths):.2f} steps")

    print(f"Total successes: {successes}, Total failures: {failures}")
    # print(f"Number of collected samples: {sum([x.shape[0] for x in x_traj])}")
    print(f"Number of collected pairs: {x_traj.shape[0]}")

if __name__ == "__main__":
    main()
