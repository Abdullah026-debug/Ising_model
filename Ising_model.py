import Glauber_functions as gb
import matplotlib
import time
matplotlib.use('TKAgg')

import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

start_time = time.time()
# Parameters
J = 1.0
nstep = 10000

# Input
if len(sys.argv) != 4:
    print("Ising_model.py (size,temperature) N T")
    sys.exit()

lx = int(sys.argv[1])
ly = lx
kT = float(sys.argv[2])
Name = str(sys.argv[3])

# Initialize spins randomly
spin = gb.random_lattice(lx)

# Create a custom colormap for yellow (-1) and purple (+1)
colors = ['yellow', 'purple']  # Specify the colors
cmap = ListedColormap(colors)

# Function to perform one Glauber dynamics step
def update(frame):
    global spin
    for _ in range(lx * ly):  # Perform a full lattice sweep
        itrial = np.random.randint(0, lx)
        jtrial = np.random.randint(0, ly)

        # Calculate ΔE and Metropolis acceptance criterion
        deltaE = -2.0 * gb.Energy_of_state_I(spin, itrial, jtrial, lx)
        accept_criterion = gb.Metropolis(deltaE, kT)
        
        # Metropolis test
        if random.random() < accept_criterion:
            spin[itrial, jtrial] *= -1

    # Update the plot
    im.set_array(spin)
    ax.set_title(f"Time Step: {frame + 1}", fontsize=12)
    return [im, ax.title] 

# Set up the figure and animation
fig, ax = plt.subplots()
im = ax.imshow(spin, animated=True, cmap=cmap, interpolation='nearest')
ani = animation.FuncAnimation(fig, update, frames=nstep, interval=100, blit=True)
ani.save(Name, writer="ffmpeg", fps=200)
End_time = time.time()

Running_time = End_time - start_time


print(f"Total runtime of the script: {Running_time:.2f} seconds")
print("Animation saved successfully.")

# Show the animation
#plt.show()


