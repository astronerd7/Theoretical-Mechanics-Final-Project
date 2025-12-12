import numpy as np
import cupy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import pyvista as pv
from matplotlib.animation import FuncAnimation
import scipy.stats as stats


#Define Constants
Ng = 128
R_bound = 1.0

mesh1d = cp.linspace(0, R_bound, Ng, endpoint=False)
mesh = cp.stack(
    cp.meshgrid(
        mesh1d, mesh1d, mesh1d, indexing='ij'
    )
)
dr = R_bound/Ng

wavenumbers_1d_real = cp.fft.rfftfreq(Ng, 1/Ng) * 2 * np.pi /R_bound
wavenumbers_1d_full = cp.fft.fftfreq(Ng, 1/Ng)*2*np.pi/R_bound

wavenumbers = cp.stack(
    cp.meshgrid(
        wavenumbers_1d_full, wavenumbers_1d_full, wavenumbers_1d_real, indexing='ij'
    )
)

wx2, wy2, wz2 = cp.meshgrid(wavenumbers_1d_full**2, wavenumbers_1d_full**2, wavenumbers_1d_real**2, indexing='ij')
#w2 = wx2+wy2+wz2 
#w2[0, 0, 0] = 1
w2 = (2/dr**2) * (3 - cp.cos(wavenumbers[0]*dr) - cp.cos(wavenumbers[1]*dr) - cp.cos(wavenumbers[2]*dr))
w2[0,0,0] = 1   # avoid dividing by zero


derivative_operator = 1j/dr * cp.sin(wavenumbers*dr)

#Change this number to start the simulation at a different time
z_int = 500
z_f = 0

omega_m = 0.3
omega_l = 0.7

class Simulation():
    def __init__(self, R0, V0, m, N, deltat, nsteps, G=1, model="matter"):
        self.R = R0
        self.V = V0
        self.m = m  
        self.N = N 
        self.G = G
        self.dt = deltat
        self.nsteps = nsteps
        self.densityTemplate = cp.zeros((Ng, Ng, Ng))
        self.a = 1/(z_int+1) 
        self.model = model
        self.H0 = 1
    
    def update_a(self, t):
        if self.model=="matter":
            a_new = (3/2 *self.H0*t)**(2/3)

        if self.model=="radiation":
            a_new = (2*self.H0*t)**(1/2)
        
        if self.model=="lambda":
            a_new = cp.exp(t*self.H0)

        return a_new
    
    def t_a(self, a):
        if self.model=="matter":
            t_new = (2/(3*self.H0))*a**(3/2)

        if self.model=="radiation":
            t_new = (1/(2*self.H0))*a**2
        
        if self.model=="lambda":
            t_new = cp.log(a)/self.H0

        return t_new

    def z_a(self, a):
        return (1/a)-1
    
    def a_z(self, z):
        return 1/(1+z)
    
    def H_t(self, t):
        if self.model=="matter":
            H = 2/(3*t)

        if self.model=="radiation":
            H = 1/(2*t)

        if self.model=="lambda":
            H = self.H0
        
        return H


    def assign_grid(self, R):
        #Create empty map
        densityMap = self.densityTemplate.copy()
        densityMap.fill(0)

        #Get particle coordinates
        rGrid = cp.rint(R / dr).astype(int)
        rGrid %= Ng 
        #Fill grid with mass
        cp.add.at(densityMap, (rGrid[:,0], rGrid[:,1], rGrid[:,2]), self.m / dr**3)

        return densityMap, rGrid
    
    def solve_poisson(self, R, t):
        #Grab grid from position
        a = self.update_a(t)
        densityMap, rGrid = self.assign_grid(R)
        meanDensity = cp.mean(densityMap)
        delta = densityMap-meanDensity
        transformed_density = cp.fft.rfftn(delta)
        phi_k = -4*cp.pi*(a**2)*self.G*transformed_density/w2
        phi_k[0, 0, 0] = 0

        #grad_phi_k = derivative_operator * phi_k 
        accelerationGrid = cp.zeros((3, Ng, Ng, Ng))
        for i in range(3):
            accelerationGrid[i] = -cp.fft.irfftn(derivative_operator[i]*phi_k, s=(Ng, Ng, Ng))/a**2

        accelArray = cp.zeros_like(R)

        accelArray = accelerationGrid[:, rGrid[:,0], rGrid[:,1], rGrid[:,2]].T


        return accelArray, densityMap


    def leapfrog(self, R, V, t, dt):
        A = self.solve_poisson(R, t)[0]
        H = self.H_t(t)
  
        V_half = V + 0.5*dt*(A-2*H*V)
        R_new = R+(dt*V_half)
        A_new, dmap = self.solve_poisson(R_new, t+0.5*dt)
        H_new = self.H_t(t+0.5*dt)
        V_new = V_half + 0.5*dt*(A_new-2*H_new*V_half)
        R_new %= R_bound

        return R_new, V_new, dmap

    def Simulate(self):
        a0 = self.a_z(z_int)
        t0 = self.t_a(a0)
        a_final = 1
        tf = self.t_a(a_final)

        a_output = cp.zeros(self.nsteps+1)
        a_output[0] = a0
        t_pts = np.linspace(t0, tf, num=self.nsteps)
        dt = t_pts[1]-t_pts[0]
        output_d = cp.zeros([self.nsteps+1, Ng, Ng, Ng])
        current_x = self.R
        current_v = self.V
        output_d[0] = self.assign_grid(self.R)[0]
        for timestep in range(self.nsteps):
            current_t = t_pts[timestep]
            next_x, next_v, dmap = self.leapfrog(R=current_x, V=current_v, t=current_t, dt=dt)
            next_a = self.update_a(current_t)
            current_x = next_x
            current_v = next_v
            output_d[timestep+1] = dmap
            a_output[timestep+1] = next_a

        return output_d, a_output
    

        


N=200000
#Uncomment the next 2 lines and comment the 3rd line to generate random initial conditions. I've included the initial conditions I used in the write-up in this repo.
#rint = cp.random.rand(N, 3)*R_bound
#np.savetxt("set_initial_conditions.npy", rint)
rint = np.loadtxt("set_initial_conditions.npy", dtype=float)
rint = cp.array(rint)
vint = cp.zeros((N, 3))
m = cp.ones(N) * 1/N

deltat = 0.001
nsteps = 400
Gsim = 5 #Setting G to 5 is a good balance between actually forming structure while not having everything collapse into one big ball.
startTime = time.time()
classTest = Simulation(R0=rint, V0=vint, m=m, N=N, deltat=deltat, nsteps=nsteps, G=Gsim, model="matter")
outd, a_history = classTest.Simulate()

finalD = outd[-1]
initD = outd[0]

density = cp.asnumpy(outd)

endTime = time.time()
simTime = endTime-startTime
print('Time spent simulating', simTime)
print(a_history[-1])



projFinal = cp.sum(finalD, axis=2).get()  # project along z


projInit = cp.sum(initD, axis=2).get()  # project along z

figFinal, ax = plt.subplots()
im0 = ax.imshow(projFinal, cmap="magma", norm="log")
cbar = plt.colorbar(im0, ax=ax)
cbar.set_label("Mass Density")
endA = a_history[-1]
plt.title(f"Matter Density; Final Scale Factor: {endA}")
plt.show()

"""
Uncomment this code to see the 2D density plot animated


fig2d = plt.figure(figsize=(12, 12))
ax2d = fig2d.add_subplot()
im2d = ax2d.imshow(np.sum(density[0], axis=2), cmap='magma', norm='log')
tracelength = 100
Nframes = 100
framestep = int(nsteps/Nframes)
traceFlag = False


def update2d(frame):
    data = np.sum(density[frame*framestep], axis=2)

    plt.title(f"Scale Factor: {a_history[frame*framestep]}")

    im2d.set_array(data)
    return [im2d]


figFinal, ax = plt.subplots()
ax.imshow(projFinal, cmap="magma", norm="log")
endA = a_history[-1]
plt.title(f"Final Scale Factor: {endA}")
#plt.savefig("Radiation.png")

animation = FuncAnimation(fig2d, func=update2d, frames=Nframes, interval=20)
plt.show()
"""


npix = density[-1].shape[0]

fourier_image = np.fft.fftn(density[-1])
fourier_amplitudes = np.abs(fourier_image)**2

kfreq = np.fft.fftfreq(npix) * npix
kfreq3D = np.meshgrid(kfreq, kfreq, kfreq)
knrm = np.sqrt(kfreq3D[0]**2 + kfreq3D[1]**2 + kfreq3D[2]**2)

knrm = knrm.flatten()
fourier_amplitudes = fourier_amplitudes.flatten()

kbins = np.arange(0.5, npix//2+1, 1.)
kvals = 0.5 * (kbins[1:] + kbins[:-1])
Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic = "mean",
                                     bins = kbins)
Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)


plt.loglog(kvals, Abins)
plt.xlabel("$k$")
plt.ylabel("$P(k)$")
plt.title("Matter Power Spectum")
plt.tight_layout()
#plt.ylim((10e9, 10e10))
plt.show()




# densities shape: (Nx, Ny, Nz, Nt)

# Build a uniform grid
pl = pv.Plotter()

grid = pv.wrap(density[0])

pl.open_movie("mattercollapse.mp4")

actor = pl.add_volume(grid, opacity="linear", clim=[0,80])
pl.add_text(f"Redshift: {(1/a_history[0])-1}", name="inttext")

pl.write_frame()

pl.remove_actor("inttext")

for i in range(nsteps):
    vol_prop = actor.GetMapper().GetInput()
    grid = pv.wrap(vol_prop)
    pl.add_text(f"Redshift: {(1/a_history[i])-1}", name="text")

    grid.point_data.active_scalars[:] = density[i].ravel()
    pl.write_frame()
    pl.remove_actor("text")

pl.close()

pl2 = pv.Plotter()

pl2.add_volume(density[-1], opacity="linear", clim=[0,80])

pl2.show()
