from casadi import *
import numpy as np
import matplotlib.pyplot as plt

dt = 0.005
T_max = 0.5
collective_thrust_max = 30.0
# body_rate_acc_max = body_rate_acc_max
m = 1  # mass
arm_l = 0.17
CT = 1
CM = 0.016
Gz = -9.8

dwx_max = 8.0
dwy_max = 8.0
dwz_max = 4.0

goal_vx = [-0.05, 0.05]
goal_vy = [-0.05, 0.05]
goal_vz = [-0.05, 0.05]

goal_wx = [-0.05, 0.05]
goal_wy = [-0.05, 0.05]
goal_roll = [-0.05, 0.05]
goal_pitch = [-0.05, 0.05]

u_max = np.array([collective_thrust_max,dwx_max,dwy_max,dwz_max])

state_range = np.array(
            [
                [-4, 4],
                [-4, 4],
                [-4, 4],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-10, 10],
                [-10, 10],
                [-10, 10],
                [-10, 10],
                [-10, 10],
                [-10, 10],
            ],dtype=np.float64
        )

# x_0 = np.array([0.,0,0,0.9063, 0.1604, 0.3753, -0.0660,-0.,-0.,0,0,0.,0.])
x_0 = np.array([0.,0,0,1,0.,0,0,0.75,0.75,0.0,0.,0.,0])

N = int(T_max / dt)

opti = Opti()

# ---- utils ----
def quat_to_rpy_casadi(q):
    """
    Returns [roll, pitch, yaw] such that R^{b->w} = Rx(roll) Ry(pitch) Rz(yaw)
    Intrinsic XYZ convention. Gives [-45, 0, 90] for q=[0.653,-0.271,0.271,0.653]
    q: [w, x, y, z] body-to-world quaternion
    """
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    # Rotation matrix elements of R^{b->w}
    R12 =     2*(x*y - w*z)
    R13 =     2*(x*z + w*y)
    R23 =     2*(y*z - w*x)
    R33 = 1 - 2*(x*x + y*y)
    R11 = 1 - 2*(y*y + z*z)

    roll  = atan2(-R23, R33)
    pitch = asin(R13)
    yaw   = atan2(-R12, R11)

    return vertcat(roll, pitch, yaw)

# ---- decision variables ---------
X = opti.variable(13, N + 1)  # state trajectory
U = opti.variable(4, N)  # control trajectory (throttle)



Q = np.eye(7) * 100
cost = 0

X_guess = np.tile(x_0,(N+1,1)).T

print(X_guess)

# ---- dynamic constraints --------
f = lambda x, u: vertcat(
    x[7], 
    x[8], 
    x[9], 
    -(x[10] * x[4] + x[11] * x[5] + x[12] * x[6]) / 2.0,
    (x[10] * x[3] + x[12] * x[5] - x[11] * x[6]) / 2.0,
    (x[11] * x[3] - x[12] * x[4] + x[10] * x[6]) / 2.0,
    (x[12] * x[3] + x[11] * x[4] - x[10] * x[5]) / 2.0,
    2 * (x[3] * x[5] + x[4] * x[6]) * CT / m * u[0],
    2 * (-x[3] * x[4] + x[5] * x[6]) * CT / m * u[0],
    Gz + (1 - 2 * pow(x[4], 2) - 2 * pow(x[5], 2)) * CT / m * u[0],
    (u[1]) - 5 * x[11] * x[12] / 9.0,
    (u[2]) + 5 * x[10] * x[12] / 9.0,
    (u[3])
)  # dx/dt = f(x,u)

for k in range(N): # loop over control intervals
   # Runge-Kutta 4 integration
   k1 = f(X[:,k],         U[:,k])
   k2 = f(X[:,k]+dt/2*k1, U[:,k])
   k3 = f(X[:,k]+dt/2*k2, U[:,k])
   k4 = f(X[:,k]+dt*k3,   U[:,k])
   x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
#    x_next[3:7] = x_next[3:7] / norm_2(x_next[3:7])
   opti.subject_to(X[:,k+1]==x_next) # close the gaps
   opti.set_initial(X,X_guess)
#    rpy = quat_to_rpy_casadi(X[3:7,k])
   x_cost = vertcat(X[4:6,k], X[7:12,k])
   cost += x_cost[:3].T @ Q[:3,:3] @ x_cost[:3]

#    # Quaternion unit norm constraint at each knot point
# for k in range(N+1):
#     q = X[3:7, k]
#     opti.subject_to(opti.bounded(0.98,q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2,1.02))

# ---- path constraints -----------
opti.subject_to(X[0,:]>= state_range[0,0])
opti.subject_to(X[0,:]<= state_range[0,1])
opti.subject_to(X[1,:]>= state_range[1,0])
opti.subject_to(X[1,:]<= state_range[1,1])
opti.subject_to(X[2,:]>= state_range[2,0])
opti.subject_to(X[2,:]<= state_range[2,1])

opti.subject_to(U[0,:] >=-u_max[0]) # control is limited
opti.subject_to(U[0,:] <=u_max[0])
opti.subject_to(U[1,:] >=-u_max[1]) # control is limited
opti.subject_to(U[1,:] <=u_max[1])
opti.subject_to(U[2,:] >=-u_max[2]) # control is limited
opti.subject_to(U[2,:] <=u_max[2])
opti.subject_to(U[3,:] >=-u_max[3]) # control is limited
opti.subject_to(U[3,:] <=u_max[3])


# ---- boundary conditions --------
opti.subject_to(X[:,0]==x_0)   # start at position x_0

opti.subject_to(opti.bounded(goal_vx[0],X[7,-1],goal_vx[1]))
opti.subject_to(opti.bounded(goal_vy[0],X[8,-1],goal_vy[1]))
opti.subject_to(opti.bounded(goal_vz[0],X[9,-1],goal_vz[1]))

opti.subject_to(opti.bounded(goal_wx[0],X[10,-1],goal_wx[1]))
opti.subject_to(opti.bounded(goal_wy[0],X[11,-1],goal_wy[1]))

rpy = quat_to_rpy_casadi(X[3:7,-1])

opti.subject_to(opti.bounded(goal_roll[0],rpy[0],goal_roll[1]))
opti.subject_to(opti.bounded(goal_pitch[0],rpy[1],goal_pitch[1]))

# ---- objective          ---------
opti.minimize(cost)  # just respect constraint

# ---- solve NLP              ------
# opti.solver("ipopt") # set numerical backend
# Set solver with options
opts = {
    'ipopt.tol': 1e-6,           # Overall convergence tolerance
    # 'ipopt.constr_viol_tol': 1e-6,  # Constraint violation tolerance
    # 'ipopt.dual_inf_tol': 1e-6,     # Dual infeasibility tolerance
    # 'ipopt.compl_inf_tol': 1e-6,    # Complementarity tolerance
    # 'ipopt.max_iter': 1000,         # Max iterations
    # 'ipopt.acceptable_tol': 1e-6,   # Acceptable (relaxed) tolerance
    # 'ipopt.print_level': 5,         # Verbosity (0=silent, 5=verbose)
}
opti.solver('ipopt', opts)
sol = opti.solve()   # actual solve


# ---- post-processing        ------
traj_x = sol.value(X)
traj_u = sol.value(U)

labels_x = ['x','y','z','q_w','q_x','q_y','q_z','v_x','v_y','v_z','w_x','w_y','w_z']
labels_u = ['f_tot','alpha_x','alpha_y','alpha_z']
def plot_x_trajectory(traj):
    """
    traj: (n, 13) tensor or numpy array
    """

    dim, n = traj.shape
    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    axes = axes.flatten()

    time = np.arange(n)*dt

    for i in range(dim):
        axes[i].plot(time, traj[i, :])
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

    dim, n = traj.shape
    fig, axes = plt.subplots(1, 4, figsize=(14, 10))
    axes = axes.flatten()

    time = np.arange(n)*dt

    for i in range(dim):
        axes[i].plot(time, traj[i, :])
        axes[i].set_title(f"{labels_u[i]}")
        axes[i].set_xlabel("Time")
        axes[i].grid(True)

    plt.tight_layout()

plot_x_trajectory(traj_x)
plot_u_trajectory(traj_u)

# try to apply same commands to scaled initial velocity

x=MX.sym('x',13)
u=MX.sym('u',4)
dt_step = MX.sym('dt_step',1)

k1 = f(x,         u)
k2 = f(x+dt_step/2*k1, u)
k3 = f(x+dt_step/2*k2, u)
k4 = f(x+dt_step*k3,   u)
x_next = x + dt_step/6*(k1+2*k2+2*k3+k4) 

f_rk4 = Function('rk4_step',[x,u,dt_step],[x_next])


# integrated trajectory
alpha=0.6
x0_scaled = x_0
x0_scaled[7:10] *= alpha 
x_int = np.zeros(traj_x.shape)
x_int[:,0] = x0_scaled
u_seq = np.zeros(traj_u.shape)
for i in range(traj_u.shape[1]):
    u_scaled =np.copy(traj_u[:,i])
    u_scaled[0] *= alpha
    x_int[:,i+1] = np.array(f_rk4(x_int[:,i],u_scaled,dt)).squeeze()
    u_seq[:,i]= np.array(u_scaled).squeeze()

# print(np.array(x_int).squeeze())
plot_x_trajectory(x_int)
plot_u_trajectory(u_seq)
# plt.show()

from scipy.spatial.transform import Rotation as ROT


def colored_3d_trajectory(ax, x, y, z,quat, cmap="plasma", lw=2, step=20, arrow_len = 0.10):
    """Plot a 3D trajectory with color scaled by time (index along path)."""
    t = np.arange(traj_x.shape[1])*dt
    norm = plt.Normalize(0, 1)
    colors = plt.get_cmap(cmap)(norm(t[:-1]))

    # Build and draw one segment per consecutive point pair
    points = np.array([x, y, z]).T
    for i, col in enumerate(colors):
        ax.plot(points[i:i+2, 0], points[i:i+2, 1], points[i:i+2, 2],
                color=col, linewidth=lw, solid_capstyle="round")
        
    # --- pose arrows ---
    for i in np.arange(0, len(x), step):
        pos = np.array([x[i], y[i], z[i]])
        quat_ = np.hstack([quat[1:,i],quat[0,i]])
        R = ROT.from_quat(quat_).as_matrix()
        for col, axis_idx in zip(["red", "green", "blue"], [0, 1, 2]):
            d = R[:, axis_idx] * arrow_len
            ax.quiver(pos[0], pos[1], pos[2],
                      d[0], d[1], d[2],
                      color=col, linewidth=1.2, arrow_length_ratio=0.3)
    
    # --- unitary aspect ratio ---
    all_pts = np.array([x, y, z])
    max_range = (all_pts.max(axis=1) - all_pts.min(axis=1)).max() / 2
    mid = all_pts.mean(axis=1)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_box_aspect([1, 1, 1])

    return norm, cmap

def add_colorbar(fig, ax, norm, cmap):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.12, shrink=0.6, aspect=20)
    cbar.set_label("Time", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Start", "Mid", "End"])
    cbar.ax.tick_params(labelsize=8)

fig = plt.figure()
fig.suptitle("3D Trajectories — color encodes time", fontsize=13, fontweight="bold")

ax1 = fig.add_subplot(111, projection="3d")

x1, y1, z1, quat1 = traj_x[0,:], traj_x[1,:], traj_x[2,:], traj_x[3:7,:]
norm1, cmap1 = colored_3d_trajectory(ax1, x1, y1, z1,quat1, cmap="turbo")

ax1.set_title("Original traj", fontsize=11)
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
ax1.legend(fontsize=8, loc="upper left")
ax1.view_init(elev=25, azim=45)

# x2, y2, z2, quat2 = x_int[0,:], x_int[1,:], x_int[2,:], x_int[3:7,:]
# norm2, cmap2 = colored_3d_trajectory(ax1, x2, y2, z2, quat2, cmap="cool")

# ax1.set_title("Scaled control trajectory", fontsize=11)
# ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
# ax1.legend(fontsize=8, loc="upper left")
# ax1.view_init(elev=25, azim=45)
# # add_colorbar(fig, ax2, norm2, cmap2)

plt.tight_layout()
plt.savefig("3d_trajectories.png", dpi=150, bbox_inches="tight")
plt.show()



def get_opti(x_init):
    opti = Opti()

    # ---- decision variables ---------
    X = opti.variable(13, N + 1)  # state trajectory
    U = opti.variable(4, N)  # control trajectory (throttle)

    Q = np.eye(7) * 100
    cost = 0

    X_guess = np.tile(x_0,(N+1,1)).T

    print(X_guess)

    # ---- dynamic constraints --------
    f = lambda x, u: vertcat(
        x[7], 
        x[8], 
        x[9], 
        -(x[10] * x[4] + x[11] * x[5] + x[12] * x[6]) / 2.0,
        (x[10] * x[3] + x[12] * x[5] - x[11] * x[6]) / 2.0,
        (x[11] * x[3] - x[12] * x[4] + x[10] * x[6]) / 2.0,
        (x[12] * x[3] + x[11] * x[4] - x[10] * x[5]) / 2.0,
        2 * (x[3] * x[5] + x[4] * x[6]) * CT / m * u[0],
        2 * (-x[3] * x[4] + x[5] * x[6]) * CT / m * u[0],
        Gz + (1 - 2 * pow(x[4], 2) - 2 * pow(x[5], 2)) * CT / m * u[0],
        (u[1]) - 5 * x[11] * x[12] / 9.0,
        (u[2]) + 5 * x[10] * x[12] / 9.0,
        (u[3])
    )  # dx/dt = f(x,u)

    for k in range(N): # loop over control intervals
        # Runge-Kutta 4 integration
        k1 = f(X[:,k],         U[:,k])
        k2 = f(X[:,k]+dt/2*k1, U[:,k])
        k3 = f(X[:,k]+dt/2*k2, U[:,k])
        k4 = f(X[:,k]+dt*k3,   U[:,k])
        x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
        #    x_next[3:7] = x_next[3:7] / norm_2(x_next[3:7])
        opti.subject_to(X[:,k+1]==x_next) # close the gaps
        opti.set_initial(X,X_guess)
        #    rpy = quat_to_rpy_casadi(X[3:7,k])
        x_cost = vertcat(X[4:6,k], X[7:12,k])
        cost += x_cost[:3].T @ Q[:3,:3] @ x_cost[:3]

        #    # Quaternion unit norm constraint at each knot point
        # for k in range(N+1):
        #     q = X[3:7, k]
        #     opti.subject_to(opti.bounded(0.98,q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2,1.02))

    # ---- path constraints -----------
    opti.subject_to(X[0,:]>= state_range[0,0])
    opti.subject_to(X[0,:]<= state_range[0,1])
    opti.subject_to(X[1,:]>= state_range[1,0])
    opti.subject_to(X[1,:]<= state_range[1,1])
    opti.subject_to(X[2,:]>= state_range[2,0])
    opti.subject_to(X[2,:]<= state_range[2,1])

    opti.subject_to(U[0,:] >=-u_max[0]) # control is limited
    opti.subject_to(U[0,:] <=u_max[0])
    opti.subject_to(U[1,:] >=-u_max[1]) # control is limited
    opti.subject_to(U[1,:] <=u_max[1])
    opti.subject_to(U[2,:] >=-u_max[2]) # control is limited
    opti.subject_to(U[2,:] <=u_max[2])
    opti.subject_to(U[3,:] >=-u_max[3]) # control is limited
    opti.subject_to(U[3,:] <=u_max[3])


    # ---- boundary conditions --------
    opti.subject_to(X[:,0]==x_init)   # start at position x_0

    opti.subject_to(opti.bounded(goal_vx[0],X[7,-1],goal_vx[1]))
    opti.subject_to(opti.bounded(goal_vy[0],X[8,-1],goal_vy[1]))
    opti.subject_to(opti.bounded(goal_vz[0],X[9,-1],goal_vz[1]))

    opti.subject_to(opti.bounded(goal_wx[0],X[10,-1],goal_wx[1]))
    opti.subject_to(opti.bounded(goal_wy[0],X[11,-1],goal_wy[1]))

    rpy = quat_to_rpy_casadi(X[3:7,-1])

    opti.subject_to(opti.bounded(goal_roll[0],rpy[0],goal_roll[1]))
    opti.subject_to(opti.bounded(goal_pitch[0],rpy[1],goal_pitch[1]))

    # ---- objective          ---------
    opti.minimize(cost)  # just respect constraint

    # ---- solve NLP              ------
    # opti.solver("ipopt") # set numerical backend
    # Set solver with options
    opts = {
        'ipopt.tol': 1e-6,           # Overall convergence tolerance
        # 'ipopt.constr_viol_tol': 1e-6,  # Constraint violation tolerance
        # 'ipopt.dual_inf_tol': 1e-6,     # Dual infeasibility tolerance
        # 'ipopt.compl_inf_tol': 1e-6,    # Complementarity tolerance
        # 'ipopt.max_iter': 1000,         # Max iterations
        # 'ipopt.acceptable_tol': 1e-6,   # Acceptable (relaxed) tolerance
        # 'ipopt.print_level': 5,         # Verbosity (0=silent, 5=verbose)
    }
    opti.solver('ipopt', opts)
    return opti

states_to_test = np.load("sampled_states.npy")
failures = []
for i in range(states_to_test.shape[0]):
    # print(f"Testing state {i+1}/{states_to_test.shape[0]}: {states_to_test[i]}")
    opti = get_opti(states_to_test[i])
    try:
        sol = opti.solve()   # actual solve
    except:
        print(f"Failed to solve for state {i+1}")
        failures.append(i)

print(f'Number of failures: {len(failures)}/{states_to_test.shape[0]}')