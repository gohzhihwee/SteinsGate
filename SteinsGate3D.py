import numpy as np
import matplotlib as mpl
 
import matplotlib.colors as colors
import matplotlib.cm as cmx
 
from scipy import integrate
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
 
plt.close('all')
 
jet = cm = plt.get_cmap('jet') 
values = range(10)
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
 
def solve_lorenz(N=12, angle=0.0, max_time=8.0, sigma=10.0, beta=8./3, rho=28.0):
 
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.axis('off')
 
    # prepare the axes limits
    ax.set_xlim((-6, 6))
    ax.set_ylim((-6, 6))
    ax.set_zlim((-55, 55))
 
    def flow_deriv(x_y_z, t0, z0=25, sigma=sigma, beta=beta, rho=rho):
        """Compute the time-derivative of a Lorenz system."""
        x, y, z = x_y_z
        R2 = x**2 + y**2 + (z-z0)**2
        R = np.sqrt(R2)
        arg = (R-2)/0.1
        env1 = 1/(1+np.exp(arg)) #Heaviside function H(r-r_0)
        env2 = 1 - env1 #H(r_0-r) = 1-H(r-r_0)

        f = env2*(x*(1/(R-1.99)**2 + 1e-2) + sigma*(y-x)) + env1*(y - (z-z0) + x*(1 - R))  #xdiff
        g = env2*(y*(1/(R-1.99)**2 + 1e-2) - y - rho*x - x*z) + env1*((z-z0) - x + y*(1 - R)) #ydiff
        h = env2*((z-z0)*(1/(R-1.99)**2 + 1e-2) + x*y - beta*z) + env1*(x - y + (z-z0)*(1 - R)) #zdiff
        return [f,g,h]
 
    # Choose random starting points, uniformly distributed from -15 to 15
    np.random.seed(1)
    x0 = -10 + 20 * np.random.random((N, 3))
 
    # Solve for the trajectories
    t = np.linspace(0, max_time, int(500*max_time))
    x_t = np.asarray([integrate.odeint(flow_deriv, x0i, t)
                      for x0i in x0])
 
    # choose a different color for each trajectory
    # colors = plt.cm.viridis(np.linspace(0, 1, N))
    # colors = plt.cm.rainbow(np.linspace(0, 1, N))
    # colors = plt.cm.spectral(np.linspace(0, 1, N))
    colors = plt.cm.prism(np.linspace(0, 1, N))
 
    for i in range(N):
        x, y, z = x_t[i,:,:].T
        lines = ax.plot(x, y, z, '-', c=colors[i])
        plt.setp(lines, linewidth=1)
 
    ax.view_init(30, angle)
    plt.show()
 
    return t, x_t
 
 
t, x_t = solve_lorenz(angle=0, N=12)
 
plt.figure(2)
lines = plt.plot(t,x_t[1,:,0],t,x_t[1,:,1],t,x_t[1,:,2])
plt.setp(lines, linewidth=1)
lines = plt.plot(t,x_t[2,:,0],t,x_t[2,:,1],t,x_t[2,:,2])
plt.setp(lines, linewidth=1)
lines = plt.plot(t,x_t[10,:,0],t,x_t[10,:,1],t,x_t[10,:,2])
plt.setp(lines, linewidth=1)

