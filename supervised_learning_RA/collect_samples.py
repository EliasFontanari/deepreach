import numpy as np
from quad_ocp import QuadOCP
import matplotlib.pyplot as plt
import tqdm as tqdm

n_episodes = 1000
MPC_frequency = 10
T_tot = 7.5


solver = QuadOCP()

print(f'Timesteps = {T_tot/solver.dt}')

rollout_states = []
l_s, g_s = [], []

successes = 0
failures = 0
for episode in tqdm.tqdm(range(n_episodes)):
    stop_episodes = False
    x0 = np.random.uniform(low=solver.state_range[:, 0], high=solver.state_range[:, 1])
    x0[3:7] /= np.linalg.norm(x0[3:7])

    print(f'Initial state: {x0}')
    rollout_states.append(x0)
    l_s.append(solver.compute_l(x0))
    g_s.append(solver.compute_g(x0))
    for i in range(int(T_tot / solver.dt)):
        if i % MPC_frequency == 0:
            sol_u = solver.solve(rollout_states[-1])["traj_u"][:, :MPC_frequency]
        x_next = np.array(solver.rk4_step_fn()(rollout_states[-1], sol_u[:, i % MPC_frequency])).squeeze()  
        l_s.append(solver.compute_l(x_next))
        g_s.append(solver.compute_g(x_next))        
        rollout_states.append(x_next)

        # print(f"Is episode {episode}, step {i}: State {x_next} within target region? {l_s[-1]}, Constraint violation: {g_s[-1]}")

        # if i % 50 == 0 and i > 0:
        #     plot = solver.plot_xy_trajectory(np.array(rollout_states[-i:]).T)
        #     print(f'l(x) reaching = {solver.compute_l(x_next)}')
        #     plt.show()
        if not solver.check_room_constraint(x_next[:3]) or solver.wall_fn_single(x_next[:3]) < -0.0:
            # print(f"Episode {episode}, step {i}: State {x_next[:3]} violates constraints. Ending episode.")
            for j in range(min(10,i)):
                g_s[-j] = 1 - 0.2 * j
            stop_episodes = True
            failures += 1
            break
        if l_s[-1] < 0:
            # print(f"Episode {episode}, step {i}: State {x_next[:3]} is within target region. Ending episode.")
            stop_episodes = True
            successes += 1
        if stop_episodes:
            break

print(f"Total successes: {successes}, Total failures: {failures}")
x_traj = np.array(rollout_states)
l_s = np.array(l_s)
g_s = np.array(g_s)

print(f'Number of collected samples: {x_traj.shape[0]}')
np.savez('quad_samples.npz', states=x_traj, l_s=l_s, g_s=g_s)