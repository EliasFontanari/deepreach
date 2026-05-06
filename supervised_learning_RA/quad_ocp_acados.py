import os
import sys

import casadi as cs
import numpy as np
import matplotlib.pyplot as plt


from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver 

class QuadOCPAcados:
    def __init__(
        self,
        dt=0.01,
        T_max=0.15,
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
    ):
        self.dt = dt
        self.T_max = T_max
        self.collective_thrust_max = collective_thrust_max
        self.m = m
        self.arm_l = arm_l
        self.CT = CT
        self.CM = CM
        self.Gz = Gz
        self.dwx_max = dwx_max
        self.dwy_max = dwy_max
        self.dwz_max = dwz_max
        self.corridor_width_x = corridor_width_x
        self.corridor_width_y = corridor_width_y

        self.goal_vx = (-0.05, 0.05)*10
        self.goal_vy = (-0.05, 0.05)*10
        self.goal_vz = (-0.05, 0.05)*10
        self.goal_wx = (-0.05, 0.05)*10
        self.goal_wy = (-0.05, 0.05)*10
        self.goal_roll = (-0.05, 0.05)*10
        self.goal_pitch = (-0.05, 0.05)*10

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

        self.N = int(self.T_max / self.dt)

        self.ocp = self._build_ocp()
        self.solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")

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

    @staticmethod
    def quat_to_rpy_numpy(q):
        w, x, y, z = q
        R12 = 2 * (x * y - w * z)
        R13 = 2 * (x * z + w * y)
        R23 = 2 * (y * z - w * x)
        R33 = 1 - 2 * (x * x + y * y)
        R11 = 1 - 2 * (y * y + z * z)
        roll = np.arctan2(-R23, R33)
        pitch = np.arcsin(R13)
        yaw = np.arctan2(-R12, R11)
        return np.array([roll, pitch, yaw], dtype=np.float64)

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

    def wall_fn(self, state_xyz):
        x = state_xyz[0]
        y = state_xyz[1]
        z = state_xyz[2]
        x_term = cs.power(x / (self.corridor_width_x / 2.0), 10)
        y_term = cs.power((y - 0.8) / (self.corridor_width_y / 2.0), 10)
        exp_term = cs.power(x_term + y_term, 4)
        return z - (10.0 * cs.exp(-exp_term) - 4.0)

    def _build_ocp(self):
        ocp = AcadosOcp()

        model = AcadosModel()
        model.name = "quad_ocp"

        x = cs.SX.sym("x", 13)
        u = cs.SX.sym("u", 4)
        xdot = cs.SX.sym("xdot", 13)

        f_expl = self.dynamics(x, u)
        model.x = x
        model.u = u
        model.xdot = xdot
        model.f_expl_expr = f_expl

        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.T_max

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        ocp.dims.ny = 7
        ocp.dims.ny_e = 3
        ocp.cost.Vx = np.zeros((ocp.dims.ny, 13))
        ocp.cost.Vx[0,0] = 1
        ocp.cost.Vx[1,1] = 1
        ocp.cost.Vx[2,2] = 1
        ocp.cost.Vu = np.zeros((ocp.dims.ny, 4))
        ocp.cost.Vu[3,0] = 1
        ocp.cost.Vu[4,1] = 1
        ocp.cost.Vu[5,2] = 1    
        ocp.cost.Vu[6,3] = 1
        ocp.cost.W = np.eye(ocp.dims.ny)
        ocp.cost.W[0, 0] = 1000.0
        ocp.cost.W[1, 1] = 1000.0
        ocp.cost.W[2, 2] = 1000.0    
        ocp.cost.W[3, 3] = 1.0
        ocp.cost.W[4, 4] = 1.0
        ocp.cost.W[5, 5] = 1.0
        ocp.cost.W[6, 6] = 1.0
        ocp.cost.yref = np.zeros(ocp.dims.ny)
        ocp.cost.yref[:3] = self.target_pos[:3]
        ocp.cost.Vx_e = np.zeros((3, 13))
        ocp.cost.Vx_e[0,0] = 1
        ocp.cost.Vx_e[1,1] = 1
        ocp.cost.Vx_e[2,2] = 1
        ocp.cost.W_e = np.eye(3) * 1000
        ocp.cost.yref_e = self.target_pos[:3]

        ocp.constraints.lbu = -self.u_max
        ocp.constraints.ubu = self.u_max
        ocp.constraints.idxbu = np.arange(4)

        ocp.constraints.lbx = self.state_range[:3, 0]
        ocp.constraints.ubx = self.state_range[:3, 1]
        ocp.constraints.idxbx = np.arange(3)

        ocp.constraints.idxbx_0 = np.arange(13)
        ocp.constraints.lbx_0 = np.zeros(13)
        ocp.constraints.ubx_0 = np.zeros(13)

        wall_expr = self.wall_fn(x[0:3])
        ocp.model.con_h_expr = wall_expr
        ocp.constraints.lh = np.array([0.0])
        ocp.constraints.uh = np.array([1.0e3])

        term_idx = np.array([7, 8, 9, 10, 11, 4, 5])
        ocp.constraints.idxbx_e = term_idx
        ocp.constraints.lbx_e = np.array(
            [
                self.goal_vx[0],
                self.goal_vy[0],
                self.goal_vz[0],
                self.goal_wx[0],
                self.goal_wy[0],
                self.goal_roll[0],
                self.goal_pitch[0],
            ]
        )
        ocp.constraints.ubx_e = np.array(
            [
                self.goal_vx[1],
                self.goal_vy[1],
                self.goal_vz[1],
                self.goal_wx[1],
                self.goal_wy[1],
                self.goal_roll[1],
                self.goal_pitch[1],
            ]
        )

        # Soft constraints via slacks
        # ocp.constraints.idxsbx = np.arange(3)
        ocp.constraints.idxsh = np.array([0])
        ocp.constraints.idxsbx_e = np.arange(7)
        ocp.cost.Zl = np.array([20000.0])
        ocp.cost.Zu = np.array([20000.0])
        ocp.cost.zl = np.array([0.0])
        ocp.cost.zu = np.array([0.0])
        ocp.cost.Zl_e = np.full(7, 50.0)
        ocp.cost.Zu_e = np.full(7, 50.0)
        ocp.cost.zl_e = np.zeros(7)
        ocp.cost.zu_e = np.zeros(7)
        ocp.cost.Zl_e[:4] = 0
        ocp.cost.Zu_e[:4] = 0

        # rpy = self.quat_to_rpy_casadi(x[3:7])
        # con_h_e = cs.vertcat(
        #     x[7],
        #     x[8],
        #     x[9],
        #     x[10],
        #     x[11],
        #     rpy[0],
        #     rpy[1],
        #     # cs.norm_2(x[0:2] - self.target_pos[:2]),
        # )
        # ocp.model.con_h_expr_e = con_h_e
        # ocp.constraints.lh_e = np.array(
        #     [
        #         self.goal_vx[0],
        #         self.goal_vy[0],
        #         self.goal_vz[0],
        #         self.goal_wx[0],
        #         self.goal_wy[0],
        #         self.goal_roll[0],
        #         self.goal_pitch[0],
        #         # 0.0,
        #     ]
        # )
        # ocp.constraints.uh_e = np.array(
        #     [
        #         self.goal_vx[1],
        #         self.goal_vy[1],
        #         self.goal_vz[1],
        #         self.goal_wx[1],
        #         self.goal_wy[1],
        #         self.goal_roll[1],
        #         self.goal_pitch[1],
        #         # 10,
        #     ]
        # )

        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_max_iter = 10000

        return ocp

    def solve(self, x_init):
        x_init = np.array(x_init, dtype=np.float64)
        self.solver.constraints_set(0, "lbx", x_init)
        self.solver.constraints_set(0, "ubx", x_init)
        status = self.solver.solve()

        if status != 0:
            raise RuntimeError(f"acados returned status {status}")

        traj_x = np.zeros((13, self.N + 1))
        traj_u = np.zeros((4, self.N))
        for k in range(self.N + 1):
            traj_x[:, k] = self.solver.get(k, "x")
        for k in range(self.N):
            traj_u[:, k] = self.solver.get(k, "u")

        return {
            "status": status,
            "traj_x": traj_x,
            "traj_u": traj_u,
        }

    def debug_solve(self, x_init):
        x_init = np.array(x_init, dtype=np.float64)
        self.solver.constraints_set(0, "lbx", x_init)
        self.solver.constraints_set(0, "ubx", x_init)
        status = self.solver.solve()

        stats = None
        try:
            stats = self.solver.get_stats("stat")
        except Exception:
            stats = None

        traj_x = np.zeros((13, self.N + 1))
        for k in range(self.N + 1):
            traj_x[:, k] = self.solver.get(k, "x")

        lbx = self.state_range[:, 0]
        ubx = self.state_range[:, 1]
        lower_violation = np.max(lbx[:, None] - traj_x)
        upper_violation = np.max(traj_x - ubx[:, None])

        xN = traj_x[:, -1]
        quat_xy = xN[4:6]
        target_dist = np.linalg.norm(xN[:2] - self.target_pos[:2])

        term_vals = {
            "vx": xN[7],
            "vy": xN[8],
            "vz": xN[9],
            "wx": xN[10],
            "wy": xN[11],
            "qx": quat_xy[0],
            "qy": quat_xy[1],
            "target_dist": target_dist,
        }

        return {
            "status": status,
            "stats": stats,
            "traj_x": traj_x,
            "state_bound_violation": {
                "lower_max": float(lower_violation),
                "upper_max": float(upper_violation),
            },
            "terminal_values": term_vals,
        }

    def plot_xy_trajectory(self, traj_x, ax=None, title="XY trajectory (top-down)", show_wall=True):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        ax.plot(traj_x[0, :], traj_x[1, :], linewidth=2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(self.state_range[0, 0], self.state_range[0, 1])
        ax.set_ylim(self.state_range[1, 0], self.state_range[1, 1])
        if show_wall:
            x_min, x_max = self.state_range[0, 0], self.state_range[0, 1]
            y_min, y_max = self.state_range[1, 0], self.state_range[1, 1]
            x_grid = np.linspace(x_min, x_max, 200)
            y_grid = np.linspace(y_min, y_max, 200)
            Xg, Yg = np.meshgrid(x_grid, y_grid)
            x_term = (Xg / (self.corridor_width_x / 2.0)) ** 10
            y_term = ((Yg - 0.8) / (self.corridor_width_y / 2.0)) ** 10
            exp_term = (x_term + y_term) ** 4
            wall_z = 10.0 * np.exp(-exp_term) - 4.0
            ax.contour(Xg, Yg, wall_z, levels=[0.0], colors="black", linewidths=1.0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        return ax

# x0 = np.array([2, -2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
# a = QuadOCPAcados()
# # dbg = a.debug_solve(x0)
# # print(dbg["status"])
# # print(dbg["state_bound_violation"])
# # print(dbg["terminal_values"])
# # print(dbg["stats"])
# a.plot_xy_trajectory(a.solve(x0)['traj_x'])
# plt.show()

# # check wall at x0
# x0 = np.array([2,2,0,
#                1,0,0,0,
#                0,0,0,
#                0,0,0], dtype=float)
# # wall_val = float(QuadOCPAcados().wall_fn(cs.DM(x0[0:3])))
# # print("wall_fn(x0) =", wall_val)

# # check bounds at x0
# ocp = QuadOCPAcados()
# # print("state bounds violations:", np.minimum(x0 - ocp.state_range[:,0], 0).min(),
# #       np.minimum(ocp.state_range[:,1] - x0, 0).min())

# # dbg = ocp.debug_solve(x0)
# # print(dbg["status"])
# # print(dbg["state_bound_violation"])
# # print(dbg["terminal_values"])

# ocp.plot_xy_trajectory(ocp.solve(x0)['traj_x'])
# plt.show()


ocp = QuadOCPAcados()
x0 = np.array([-0.6, -3.5, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

# # Wall value at x0 should be >= 0 to be feasible
# print("wall_fn(x0) =", float(ocp.wall_fn(x0[:3])))

# traj = ocp.solve(x0)['traj_x']
# ocp.plot_xy_trajectory(traj)
# plt.show()
# # dbg = ocp.debug_solve(x0)
# # print("status:", dbg["status"])
# # print("terminal:", dbg["terminal_values"])
# # print("state bounds:", dbg["state_bound_violation"])

# print(f'Trajectory: {traj[:2,:]}')

n_step = 140
# MPC
traj = np.zeros(( n_step+1, 13))
x_current = x0.copy()
traj[0] = x_current
for i in range(n_step):
    print(f"Step {i} time {i*ocp.dt:.3f}")
    sol = ocp.solve(traj[i])
    if sol["status"] != 0:
        print(f"Solver failed at step {i} with status {sol['status']}")
        break
    u_opt = sol["traj_u"][:, 0]
    x_next = sol["traj_x"][:, 1]
    traj[i+1] = x_next

ocp.plot_xy_trajectory(traj.T)
plt.show()


print(f'Trajectory last state: {traj[-1, :]}')
print(f'Wall value at last state: {float(ocp.wall_fn(traj[-1, :3]))}')