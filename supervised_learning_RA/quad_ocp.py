import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class QuadOCP:
    def __init__(
        self,
        dt=0.02,
        T_max=1.0,
        collective_thrust_max=30.0,
        m=1.0,
        arm_l=0.17,
        CT=1.0,
        CM=0.016,
        Gz=-9.8,
        dwx_max=8.0,
        dwy_max=8.0,
        dwz_max=4.0,
        corridor_width_x=0.5,
        corridor_width_y=8.0,
        solver_name="ipopt",
        solver_opts=None,
        soft_constraints=True,
    ):
        self.dt = dt
        self.T_max = T_max
        self.collective_thrust_max = collective_thrust_max
        self.m = m
        self.arm_l = arm_l
        self.CT = CT
        self.CM = CM
        self.Gz = Gz
        self.drone_radius = 0.12
        self.dwx_max = dwx_max
        self.dwy_max = dwy_max
        self.dwz_max = dwz_max
        self.corridor_width_x = corridor_width_x
        self.corridor_width_y = corridor_width_y
        self.solver_name = solver_name
        self.solver_opts = solver_opts
        self.l_scale = 0.5

        self.soft_constraints = soft_constraints

        self.goal_vx = (-0.05, 0.05)
        self.goal_vy = (-0.05, 0.05)
        self.goal_vz = (-0.05, 0.05)
        self.goal_wx = (-0.05, 0.05)
        self.goal_wy = (-0.05, 0.05)
        self.goal_roll = (-0.05, 0.05)
        self.goal_pitch = (-0.05, 0.05)

        self.goal_equilibrium = np.array([ self.goal_roll[1], self.goal_pitch[1],self.goal_vx[1], self.goal_vy[1], self.goal_vz[1],self.goal_wx[1], self.goal_wy[1]], dtype=np.float64)
        self.u_max = np.array(
            [self.collective_thrust_max, self.dwx_max, self.dwy_max, self.dwz_max],
            dtype=np.float64,
        )

        self.state_range = np.array(
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

        self.target_pos = np.array([2.0, -2.0, 0.0], dtype=np.float64)
        self.index_target = [4,5,7,8,9,10,11]

        self.wall_active = False
        self.cylinders_active = True

        self.cylinders = np.array([[0,2,1.2], 
                                  [0,-2, 1.2]])

        self.N = int(self.T_max / self.dt)
        self.opti, self.X, self.U, self.x_init_param = self._build_opti()

    @staticmethod
    def quat_to_rpy_casadi(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        R12 = 2 * (x * y - w * z)
        R13 = 2 * (x * z + w * y)
        R23 = 2 * (y * z - w * x)
        R33 = 1 - 2 * (x * x + y * y)
        R11 = 1 - 2 * (y * y + z * z)
        roll = cs.atan2(-R23, R33)
        pitch = cs.asin(R13)
        yaw = cs.atan2(-R12, R11)
        return cs.vertcat(roll, pitch, yaw)

    def dynamics(self, x, u):
        return cs.vertcat(
            x[7],
            x[8],
            x[9],
            -(x[10] * x[4] + x[11] * x[5] + x[12] * x[6]) / 2.0,
            (x[10] * x[3] + x[12] * x[5] - x[11] * x[6]) / 2.0,
            (x[11] * x[3] - x[12] * x[4] + x[10] * x[6]) / 2.0,
            (x[12] * x[3] + x[11] * x[4] - x[10] * x[5]) / 2.0,
            2 * (x[3] * x[5] + x[4] * x[6]) * self.CT / self.m * u[0],
            2 * (-x[3] * x[4] + x[5] * x[6]) * self.CT / self.m * u[0],
            self.Gz + (1 - 2 * cs.power(x[4], 2) - 2 * cs.power(x[5], 2)) * self.CT / self.m * u[0],
            u[1] - 5 * x[11] * x[12] / 9.0,
            u[2] + 5 * x[10] * x[12] / 9.0,
            u[3],
        )

    def rk4_step_fn(self):
        x_sym = cs.SX.sym("x", 13)
        u_sym = cs.SX.sym("u", 4)
        k1 = self.dynamics(x_sym, u_sym)
        k2 = self.dynamics(x_sym + self.dt / 2 * k1, u_sym)
        k3 = self.dynamics(x_sym + self.dt / 2 * k2, u_sym)
        k4 = self.dynamics(x_sym + self.dt * k3, u_sym)
        x_next_sym = x_sym + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        q_next_sym = x_next_sym[3:7] / cs.norm_2(x_next_sym[3:7])
        x_next_sym = cs.vertcat(x_next_sym[0:3], q_next_sym, x_next_sym[7:])
        return cs.Function("rk4_step", [x_sym, u_sym], [x_next_sym])

    def wall_fn(self, state_xyz):
        x = state_xyz[0, :]
        y = state_xyz[1, :]
        z = state_xyz[2, :]
        x_term = cs.power(x / (self.corridor_width_x / 2.0),6 )
        y_term = cs.power((y - 0.8) / (self.corridor_width_y / 2.0), 6)
        exp_term = cs.power(x_term + y_term, 4)
        return z - (10.0 * cs.exp(-exp_term) - 4.0)
    
    def wall_fn_single(self, state_xyz):
        x = state_xyz[0]
        y = state_xyz[1]
        z = state_xyz[2]
        x_term = cs.power(x / (self.corridor_width_x / 2.0),6 )
        y_term = cs.power((y - 0.8) / (self.corridor_width_y / 2.0), 6)
        exp_term = cs.power(x_term + y_term, 4)
        return z - (10.0 * cs.exp(-exp_term) - 4.0)
    
    def cylinder_fn(self, cylinder_pos, cylinder_rad, state_xyz):
        x = state_xyz[0, :]
        y = state_xyz[1, :]
        cylinder_pos = cs.repmat(cs.vertcat(cylinder_pos[0], cylinder_pos[1]), 1, x.size(2))
        xy = cs.vertcat(x, y)
        dist2_to_cylinder = cs.sum1((xy - cylinder_pos)**2)
        return dist2_to_cylinder - ((self.drone_radius + cylinder_rad) ** 2)
    
    def cylinder_fn_single(self, cylinder_pos, cylinder_rad, state_xyz):
        x = state_xyz[0]
        y = state_xyz[1]
        xy = cs.vertcat(x, y)
        dist2_to_cylinder = cs.sum1((xy - cylinder_pos)**2)
        return dist2_to_cylinder - ((self.drone_radius + cylinder_rad) ** 2)


    def check_room_constraint(self, state_xyz):
        if np.any(state_xyz > (self.state_range[:3,1] - self.drone_radius +1e-3)) or np.any(state_xyz < self.state_range[:3,0] + self.drone_radius - 1e-3):
            return 1
        return -1

    def check_reaching_target(self, state_xyz):
        if np.linalg.norm(state_xyz - self.target_pos) < 0.15 and np.all(np.abs(state_xyz[self.index_target]) < self.goal_equilibrium):
            return True
        return False
    
    def compute_l(self, state):
        pos_const = np.linalg.norm(state[:3] - self.target_pos)
        target_abs_max = np.max(np.abs(state[self.index_target]))
        raw_l = max(pos_const - 0.2, target_abs_max - 0.1)
        return np.tanh(raw_l / self.l_scale)
    
    def compute_g(self, state):
        if self.wall_active:
            wall_constr = -self.wall_fn_single(state[:3])
            if wall_constr >= 0:
                return 1
        if self.cylinders_active:
            for j_cyl in range(self.cylinders.shape[0]):
                cyl_constr = -self.cylinder_fn_single(self.cylinders[j_cyl, :2], self.cylinders[j_cyl, 2], state[:3])
                if cyl_constr >= 1e-3:
                    print(f'cylinder fail')
                    return 1
        room_constr = self.check_room_constraint(state[:3])
        if room_constr >= 0:
            print(f'room fail')
            return 1
        return -1

    def _build_opti(self):
        opti = cs.Opti()
        X = opti.variable(13, self.N + 1)
        U = opti.variable(4, self.N)
        x_init_param = opti.parameter(13)

        Q_vel = np.eye(6) * 2
        Q_target = np.eye(3) * 10
        
        cost = 0

        step_fn = self.rk4_step_fn()

        for k in range(self.N):
            x_next = step_fn(X[:, k], U[:, k])
            opti.subject_to(X[:, k + 1] == x_next)
            # x_cost = cs.vertcat(X[4:6, k], X[7:12, k])
            x_cost = X[7:, k]
            cost += x_cost.T @ Q_vel @ x_cost
            cost += (X[:3, k] - self.target_pos[:3]).T @ Q_target @ (X[:3, k] - self.target_pos[:3])


        if self.soft_constraints:
            wall_slack_weight = 50000.0
            bound_slack_weight = 50000.0
            s_pos_low = opti.variable(3, self.N + 1)
            s_pos_up = opti.variable(3, self.N + 1)
            opti.subject_to(cs.vec(s_pos_low) >= 0)
            opti.subject_to(cs.vec(s_pos_up) >= 0)
            soft_flag = 1
        else:
            soft_flag = 0
            wall_slack_weight = 0
            bound_slack_weight = 0
            s_pos_low = opti.variable(3, self.N + 1)
            s_pos_up = opti.variable(3, self.N + 1)


        opti.subject_to(X[0, :] >= self.state_range[0, 0] - soft_flag * s_pos_low[0, :] + self.drone_radius)
        opti.subject_to(X[0, :] <= self.state_range[0, 1] + soft_flag * s_pos_up[0, :] - self.drone_radius)
        opti.subject_to(X[1, :] >= self.state_range[1, 0] - soft_flag * s_pos_low[1, :] + self.drone_radius)
        opti.subject_to(X[1, :] <= self.state_range[1, 1] + soft_flag * s_pos_up[1, :] - self.drone_radius)
        opti.subject_to(X[2, :] >= self.state_range[2, 0] - soft_flag * s_pos_low[2, :] + self.drone_radius)
        opti.subject_to(X[2, :] <= self.state_range[2, 1] + soft_flag * s_pos_up[2, :] - self.drone_radius)

        opti.subject_to(U[0, :] >= 0)
        opti.subject_to(U[0, :] <= self.u_max[0])
        opti.subject_to(U[1, :] >= -self.u_max[1])
        opti.subject_to(U[1, :] <= self.u_max[1])
        opti.subject_to(U[2, :] >= -self.u_max[2])
        opti.subject_to(U[2, :] <= self.u_max[2])
        opti.subject_to(U[3, :] >= -self.u_max[3])
        opti.subject_to(U[3, :] <= self.u_max[3])

        opti.subject_to(X[:, 0] == x_init_param)


        s_cyl = opti.variable(1, self.N + 1)
        s_wall = opti.variable(1, self.N + 1)            

        if self.wall_active:
            opti.subject_to(self.wall_fn(X[0:3, :]) + soft_flag * s_wall >= 0)

        if self.cylinders_active:     
            for j_cyl in range(self.cylinders.shape[0]):
                opti.subject_to(self.cylinder_fn(self.cylinders[j_cyl, :2], self.cylinders[j_cyl, 2], X[0:2,:]) + soft_flag * s_cyl >= 0)

        # opti.subject_to(opti.bounded(self.goal_vx[0], X[7, -1], self.goal_vx[1]))
        # opti.subject_to(opti.bounded(self.goal_vy[0], X[8, -1], self.goal_vy[1]))
        # opti.subject_to(opti.bounded(self.goal_vz[0], X[9, -1], self.goal_vz[1]))

        # opti.subject_to(opti.bounded(self.goal_wx[0], X[10, -1], self.goal_wx[1]))
        # opti.subject_to(opti.bounded(self.goal_wy[0], X[11, -1], self.goal_wy[1]))

        # rpy = self.quat_to_rpy_casadi(X[3:7, -1])
        # opti.subject_to(opti.bounded(self.goal_roll[0], rpy[0], self.goal_roll[1]))
        # opti.subject_to(opti.bounded(self.goal_pitch[0], rpy[1], self.goal_pitch[1]))

        # opti.subject_to(cs.norm_2(X[:2, -1] - self.target_pos[:2]) < 0.15)

        cost +=  bound_slack_weight * cs.sumsqr(s_pos_low) + bound_slack_weight * cs.sumsqr(s_pos_up) +wall_slack_weight * cs.sumsqr(s_wall) + (wall_slack_weight if self.cylinders_active else 0) * cs.sumsqr(s_cyl)
        
        opti.minimize(
            cost  
        )

        if self.solver_opts is not None:
            opts = self.solver_opts
        elif self.solver_name == "ipopt":
            print("Using IPOPT solver with default options.")
            opts = {
                "ipopt.tol": 1e-6,
                "ipopt.print_level": 0,
                # "ipopt.start_with_resto": "yes",
                "ipopt.max_iter": 1000,
                "print_time": 0,
                "verbose": False,
            }
        else:
            opts = {}

        # opti.solver(self.solver_name, opts)
        opti.solver('fatrop', {'structure_detection':'manual', 'nx': nx, 'nu':nu, 'ng':ng, 'N':K-1, "expand": True, "fatrop.mu_init":1e-1, "jit":True, "fatrop.print_level":10, "jit_options": {"flags": "-O3", "verbose": True}})

        return opti, X, U, x_init_param

    def build_opti(self):
        return self.opti, self.X, self.U

    def solve(self, x_init):
        self.opti.set_value(self.x_init_param, x_init)
        X_guess = np.tile(x_init, (self.N + 1, 1)).T
        self.opti.set_initial(self.X, X_guess)
        sol = self.opti.solve()
        traj_x = sol.value(self.X)
        traj_u = sol.value(self.U)
        return {
            "opti": self.opti,
            "solution": sol,
            "traj_x": traj_x,
            "traj_u": traj_u,
        }

    def plot_xy_trajectory(self, traj_x, ax=None, title="XY trajectory (top-down)", show_wall=True, show_cylinders=True, show_target=True):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        x_vals = traj_x[0, :]
        y_vals = traj_x[1, :]
        if x_vals.size > 1:
            points = np.column_stack([x_vals, y_vals])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            t = np.linspace(0.0, 1.0, segments.shape[0])
            lc = LineCollection(segments, cmap="viridis", linewidths=2)
            lc.set_array(t)
            ax.add_collection(lc)
        else:
            ax.plot(x_vals, y_vals, linewidth=2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(self.state_range[0, 0], self.state_range[0, 1])
        ax.set_ylim(self.state_range[1, 0], self.state_range[1, 1])
        if show_wall and self.wall_active:
            x_min, x_max = self.state_range[0, 0], self.state_range[0, 1]
            y_min, y_max = self.state_range[1, 0], self.state_range[1, 1]
            x_grid = np.linspace(x_min, x_max, 200)
            y_grid = np.linspace(y_min, y_max, 200)
            Xg, Yg = np.meshgrid(x_grid, y_grid)
            x_term = (Xg / (self.corridor_width_x / 2.0)) ** 6
            y_term = ((Yg - 0.8) / (self.corridor_width_y / 2.0)) ** 6
            exp_term = (x_term + y_term) ** 4
            wall_z = 10.0 * np.exp(-exp_term) - 4.0
            ax.contour(Xg, Yg, wall_z, levels=[0.0], colors="black", linewidths=1.0)
        if show_cylinders and self.cylinders_active:
            for cyl in self.cylinders:
                circle = Circle((cyl[0], cyl[1]), cyl[2], color="gray", fill=False, linewidth=1.5)
                ax.add_patch(circle)
        if show_target:
            ax.scatter(self.target_pos[0], self.target_pos[1], marker="x", color="red", s=80, linewidths=2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        return ax

    def plot_xyz_trajectory(self, traj_x, ax=None, title="XYZ trajectory", show_wall=True, show_cylinders=True, show_target=True):
        if ax is None:
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
        x_vals = traj_x[0, :]
        y_vals = traj_x[1, :]
        z_vals = traj_x[2, :]
        if x_vals.size > 1:
            points = np.column_stack([x_vals, y_vals, z_vals])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            t = np.linspace(0.0, 1.0, segments.shape[0])
            lc = Line3DCollection(segments, cmap="viridis", linewidths=2)
            lc.set_array(t)
            ax.add_collection3d(lc)
        else:
            ax.plot(x_vals, y_vals, z_vals, linewidth=2)

        if show_wall and self.wall_active:
            x_min, x_max = self.state_range[0, 0], self.state_range[0, 1]
            y_min, y_max = self.state_range[1, 0], self.state_range[1, 1]
            x_grid = np.linspace(x_min, x_max, 80)
            y_grid = np.linspace(y_min, y_max, 80)
            Xg, Yg = np.meshgrid(x_grid, y_grid)
            x_term = (Xg / (self.corridor_width_x / 2.0)) ** 6
            y_term = ((Yg - 0.8) / (self.corridor_width_y / 2.0)) ** 6
            exp_term = (x_term + y_term) ** 4
            wall_z = 10.0 * np.exp(-exp_term) - 4.0
            ax.plot_surface(
                Xg,
                Yg,
                wall_z,
                rstride=1,
                cstride=1,
                color="red",
                alpha=0.25,
                linewidth=0,
                antialiased=True,
            )
        if show_cylinders and self.cylinders_active:
            z_min, z_max = self.state_range[2, 0], self.state_range[2, 1]
            theta = np.linspace(0.0, 2.0 * np.pi, 60)
            z_grid = np.linspace(z_min, z_max, 20)
            Theta, Zc = np.meshgrid(theta, z_grid)
            for cyl in self.cylinders:
                Xc = cyl[0] + cyl[2] * np.cos(Theta)
                Yc = cyl[1] + cyl[2] * np.sin(Theta)
                ax.plot_surface(
                    Xc,
                    Yc,
                    Zc,
                    rstride=1,
                    cstride=1,
                    color="gray",
                    alpha=0.2,
                    linewidth=0,
                    antialiased=True,
                )
        if show_target:
            ax.scatter(
                [self.target_pos[0]],
                [self.target_pos[1]],
                [self.target_pos[2]],
                marker="x",
                color="red",
                s=80,
                linewidths=2,
            )

        ax.set_xlim(self.state_range[0, 0], self.state_range[0, 1])
        ax.set_ylim(self.state_range[1, 0], self.state_range[1, 1])
        ax.set_zlim(self.state_range[2, 0], self.state_range[2, 1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title)
        return ax


if __name__ == "__main__":
    solver = QuadOCP()


    print(f"Compte g test: {solver.compute_g(np.array([0, -3.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0., 0., 0.0, 0.0, 0.0, 0.0], dtype=np.float64))}")
    #MPC loop
    N_max = 2000
    MPC_frequency = 10 # steps
    x0 = np.array([0.50677361, -3.54062472, -2.34912985, 0.71139237, -0.13570943, 0.68732024,
                   0.05563031, 0.45007205, 2.64777957, 0.34814984, 0.85367792, 0.19473124,
                   0.51528444], dtype=np.float64)
    # x0 = np.array([0.36829686, 3.53152368, -0.75706301, -0.28332137, 0.65517484, 0.69473076,
    #                0.08845391, 1.08366195, 2.41618241, -2.22849583, -0.78087271, -2.29923304,
    #                1.72612237], dtype=np.float64)
    solver.compute_g(x0)
    traj = np.zeros((N_max+1, 13))
    traj_u = np.zeros((N_max, 4))
    traj[0, :] = x0
    for i in range(N_max):
        if i > 0:
            print(f"Step {i+1}/{N_max}, l_reaching {solver.compute_l(x_next) } state {x_next} last u {traj_u[i-1, :]}")

        if i % MPC_frequency == 0:
            sol = solver.solve(traj[i, :])
            traj_u[i:i+MPC_frequency, :] = (sol["traj_u"][:, :MPC_frequency]).T

            traj_x = sol["traj_x"]
            wall_fails = 0
            if solver.wall_active:
                for j in range(traj_x.shape[0]):
                    if solver.wall_fn_single(traj_x[:3, j]) < 0:
                        print(f"Warning: Trajectory point {j} is violating the wall constraint with value {solver.wall_fn_single(traj_x[:3, j])} at state {traj_x[:3, j]}")
                        wall_fails += 1
                if wall_fails == 0:    
                    print("All trajectory points satisfy the wall constraint.")
            if solver.cylinders_active:
                for j_cyl in range(solver.cylinders.shape[0]):
                    cyl_fails = 0
                    for j in range(traj_x.shape[0]):
                        if solver.cylinder_fn_single(solver.cylinders[j_cyl, :2], solver.cylinders[j_cyl, 2], traj_x[:3, j]) < 0:
                            print(f"Warning: Trajectory point {j} is violating the cylinder {j_cyl} constraint with value {solver.cylinder_fn_single(solver.cylinders[j_cyl, :2], solver.cylinders[j_cyl, 2], traj_x[:3, j])} at state {traj_x[:3, j]}")
                            cyl_fails += 1
                    if cyl_fails == 0:
                        print(f"All trajectory points satisfy the cylinder {j_cyl} constraint.")
            for j in range(traj_x.shape[0]):
                if solver.check_room_constraint(traj_x[:3, j]) >= 0:
                    print(f"Warning: Trajectory point {j} is violating the room boundary constraint.")

        
        x_next = np.array(solver.rk4_step_fn()(traj[i, :], traj_u[i, :])).squeeze()
        
        # if solver.compute_l(x_next) < 0:
        #     print(f"Target reached at step {i+1} in MPC solution.")
        #     traj[i+1, :] = x_next
        #     break

        
        traj[i+1, :] = x_next
        # if i % 50 == 0:
        #     plot = solver.plot_xy_trajectory(traj[:i+1].T, title=f"MPC Trajectory (step {i+1})", show_wall=True)
        #     # plot = solver.plot_xyz_trajectory(sol["traj_x"], title=f"MPC Trajectory (step {i+1})", show_wall=True)
        #     plot = solver.plot_xyz_trajectory(traj[:i+1].T, title=f"MPC Trajectory (step {i+1})", show_wall=True)

        #     plt.show()

        #     if solver.compute_l(traj_x[:, -1]) < 0:
        #         print("Target reached in MPC solution.")
        #         break


    plot = solver.plot_xy_trajectory(traj.T, title="MPC Trajectory (top-down)", show_wall=True)
    plt.show()

    print(f'Last state: {traj[-1, :]}')



    # print(solver.solve(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.75, 0.75, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)))
    # plot = solver.plot_xy_trajectory(solver.solve(np.array([-1.0, -3.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0., 0., 0.0, 0.0, 0.0, 0.0], dtype=np.float64))["traj_x"])
    # plt.show()
    # print('solve again')
    # print(a.solve(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.75, 0.75, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)))