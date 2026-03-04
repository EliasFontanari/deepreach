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

goal_vx = [-0.1, 0.1]
goal_vy = [-0.1, 0.1]
goal_vz = [-0.1, 0.1]

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

x_0 = np.array([0.,0,0,1,0.0,0.0,0.,0.9,0.6,0,.0,0,0])
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
opti.minimize(1)  # just respect constraint

# ---- solve NLP              ------
# opti.solver("ipopt") # set numerical backend
# Set solver with options
opts = {
    'ipopt.tol': 1e-4,           # Overall convergence tolerance
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
plt.show()