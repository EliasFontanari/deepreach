import numpy as np

u_max = 1
d_max = 4
x_th = -1.5
gamma=0.99

x_min = -2.2
x_max = 2.2
grid_min = -2.1
grid_max = 3.0

target_max = -1
failure_max = -2

distr = 'gaussian'

log_repr = True

state_size = 1


def x_next(x,u,d):
    x=np.array(x)
    x_next = 1.01*x + 0.01 * (u + d)
    reached_states = np.where((failure_max <= x) & (x <= target_max))[0]
    x_next[reached_states] = x[reached_states]
    return x_next

def pi(x):
    x=np.array(x)
    u = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        if x[i] > x_th:
            u[i] = -u_max
        else:
            u[i] = u_max
    return u
    
def phi(x):
    if x >= x_th:
        return d_max
    else:
        return -d_max

def c(x):
    c = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        c[i] = np.max((np.min((x[i]-failure_max,10)),-10))
    # return np.ones(c.shape[0])
    return c

def r(x):
    r = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        r[i] = np.max((np.min((-(x[i]-target_max),10)),-10))
    return r

# noise distribution
def gen_noise():
    if distr == 'uniform':
        return np.random.uniform(-d_max,d_max,state_size)
    elif distr == 'gaussian':
        return np.random.normal(0,d_max,state_size)
    
# V_prob = np.load(f"data/V_prob_x_th:{x_th}_u_max:_{u_max}_d_max:_{d_max}_distr_{distr}.npy")
# state_grid = np.load("data/grid.npy")
    
# settings_class
class System_conf():
    def __init__(self):
        self.u_max = u_max
        self.d_max = d_max
        self.x_th = x_th
        self.gamma = gamma
        self.x_min = x_min
        self.x_max = x_max
        self.distr = distr
        self.log_repr = log_repr
        self.state_size = state_size
        self.V_prob = V_prob
        self.state_grid = state_grid