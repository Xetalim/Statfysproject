import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib import animation

np.random.seed(10)
random.seed(10)

# Sidelength of the grid
GRID_SIZE = 25

# decide whether or not to animate the simulation
ANIMATE_SIM = False
# Generating the grid, with dtype integer
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int) + 1
grid = (np.random.randint(0, 2, size=(GRID_SIZE, GRID_SIZE)) * 2) - 1

def accept_change(delta_energy, t_red):
    p = math.exp(- delta_energy / t_red)
    num = random.random()
    if p > num:
        return True
    else:
        return False
def local_energy_flipped(lattice, x_pos, y_pos):
    # doesn't have a minus sign because lattice[x,y] has been flipped
    left = lattice[y_pos, x_pos] * lattice[y_pos, (x_pos - 1) % GRID_SIZE]
    right = lattice[y_pos, x_pos] * lattice[y_pos, (x_pos + 1) % GRID_SIZE]
    top = lattice[y_pos, x_pos] * lattice[(y_pos + 1) % GRID_SIZE, x_pos]
    bottom = lattice[y_pos, x_pos] * lattice[(y_pos - 1) % GRID_SIZE, x_pos]
    # factor 2, because we don't want to count anything double
    delta_energy = 2 * (left + right + top + bottom)
    return delta_energy

def total_energy_simple(lattice):
    lattice_energy = []
    for y_pos, lattice_row in enumerate(lattice):
        lattice_row_energy = []
        for x_pos, s_i in enumerate(lattice_row):
            y_energy = lattice[(y_pos+1) % GRID_SIZE, x_pos % GRID_SIZE] + lattice[(y_pos-1) % GRID_SIZE, x_pos % GRID_SIZE]
            x_energy = lattice_row[(x_pos-1) % GRID_SIZE] + lattice_row[(x_pos+1) % GRID_SIZE]
            local_energy = -(x_energy + y_energy)*s_i
            lattice_row_energy.append(local_energy)
        lattice_energy.append(lattice_row_energy)   
    # factor 0.5 because we don't want to count double
    return 0.5 * np.sum(lattice_energy)


def total_energy_quick(lattice):
    # initialising empty energy array
    energy = np.zeros_like(lattice)
    # multiplying every spin by it's left neighbour
    energy -= np.take(lattice, np.arange(1, GRID_SIZE + 1), axis=1, mode="wrap")
    # multiplying every spin by it's right neighbour
    energy -= np.take(lattice, np.arange(-1, GRID_SIZE -1), axis=1, mode="wrap")
    # multiplying every spin by it's below neighbour
    energy -= np.take(lattice, np.arange(1, GRID_SIZE + 1), axis=0, mode="wrap")
    # multiplying every spin by it's top neighbour
    energy -= np.take(lattice, np.arange(-1, GRID_SIZE -1), axis=0, mode="wrap")

    # factor 0.5 because we don't want to count double
    return 0.5 * np.sum(energy * lattice)


def calc_stop_sim(perc, energy_list, i, index, calc_freq):
    a = np.average(energy_list[max(i // calc_freq - 50 * (i // index) // calc_freq,0):max(i // calc_freq - 25 * (i // index) // calc_freq,1)])
    b = np.average(energy_list[max(i // calc_freq - 25 * (i // index) // calc_freq,0):i // calc_freq])

    newperc = np.abs((a-b)/(a) * 100)
    min_i = max(i // calc_freq - 25 * (i // index) // calc_freq,0)
    max_i = i // calc_freq
    # print(a,b)
    # print("{} out of {}, diff = {:.2f}".format(index, maxroll, newperc))


    stop = (newperc < 2 and perc < 2)
    return stop, newperc, (min_i, max_i)


def simulate(t_red, grid, iterations, calc_freq, animate=False):
    energy = total_energy_quick(grid)
    # energy_arr = np.empty(iterations // calc_freq)
    # mag_arr = np.empty(iterations // calc_freq)
    energy_list = []
    mag_list = []
    diff_list = []
    
    perc = 100
    
    for i in range(0, iterations):
        accept = False
        # choosing random coordinates
        rand_x = int(random.random() * GRID_SIZE)
        rand_y = int(random.random() * GRID_SIZE)
        # is the same as
        # rand_x = np.random.randint(0, GRID_SIZE)
        # but about 1000 times as fast

        # calculate the difference in energy
        delta_energy = local_energy_flipped(grid, rand_x, rand_y)

        # if delta_energy is less than zero, accept the change
        # and update the energy
        if delta_energy <= 0:
            energy = energy + delta_energy
            accept = True
        # if delta_energy is more than zero, accept the change
        # with the probability as given in the algorithm
        elif accept_change(delta_energy, t_red):
            energy = energy + delta_energy
            accept = True
        if accept:
            grid[rand_y, rand_x] *= -1
        
        # calculate and append all observables
        if i % calc_freq == 0:

            energy_list.append(energy)
            mag_list.append(np.sum(grid))
            maxroll = 100
            index = (i // calc_freq) / ((iterations // calc_freq) // maxroll)
            
            # code to determine equilibrium
            # if index in np.arange(0, 50, dtype=int):
                # print(index)
            
            # check if we are in equilibrium every 100th of the simulation,
            # starting at 50
            if index in np.arange(50, 100, dtype=int):
                index = int(index)
                stopsim, perc, indices = calc_stop_sim(perc, energy_list,i, index, calc_freq)
                diff_list.append(perc)
                if stopsim:
                    break
    return energy_list, mag_list, grid, diff_list, indices

def calculate_values(en, mag, indices):
    capacity = np.var(en[indices[0]:indices[1]])
    mag = np.average(mag[indices[0]:indices[1]])
    mag_abs = np.average(np.abs(mag[indices[0]:indices[1]]))
    avg_en = np.average(en[indices[0]:indices[1]])
    return capacity, mag, mag_abs, avg_en

en, mag, endgrid, diff, indices = simulate(1, grid, int(10e4), 10)
# plotting the end grid after n iteratons
plt.imshow(endgrid, aspect='equal')
plt.xlim(-.5,GRID_SIZE-.5) #the grid starts at [-.5,-.5]
plt.ylim(-.5,GRID_SIZE-.5)
plt.show()

# animating the simulation, needs a different structure


def init():
    global energy
    global energies
    global xrange
    energy = total_energy_quick(grid)
    im.set_data(grid)
    line.set_xdata(xrange)
    line.set_ydata(energies)
    
    return [im, line]

def animate(i):
    global energy
    global grid
    global energies
    global xrange
    for j in range(0, 1000):
        accept = False
        # choosing random coordinates
        rand_x = int(random.random() * GRID_SIZE)
        rand_y = int(random.random() * GRID_SIZE)
        # is the same as
        # rand_x = np.random.randint(0, GRID_SIZE)
        # but about 1000 times as fast
    
        # calculate the difference in energy
        delta_energy = local_energy_flipped(grid, rand_x, rand_y)
    
        # if delta_energy is less than zero, accept the change
        # and update the energy
        if delta_energy <= 0:
            energy = energy + delta_energy
            accept = True
        # if delta_energy is more than zero, accept the change
        # with the probability as given in the algorithm
        elif accept_change(delta_energy, t_red):
            energy = energy + delta_energy
            accept = True
        if accept:
            grid[rand_y, rand_x] *= -1
    energies.append(energy)
    xrange.append(i)
    im.set_array(grid)
    line.set_xdata(xrange)
    line.set_ydata(energies)
    # line = [xrange, energies]
    ax[1].set_ylim(min(energies) - 100, max(energies) + 100)
    return [im, line]


if ANIMATE_SIM:
    t_red = 2
    
    fig, ax = plt.subplots(1, 2)
    # ax = plt.axes(xlim=(0, GRID_SIZE), ylim=(0, GRID_SIZE))
    # ax[0].set_xlim(0, GRID_S)
    ax[1].set_xlim(0, 1000)
    # ax[1].set_ylim(-GRID_SIZE**2, 0)
    energies = []
    xrange = []
    im=ax[0].imshow(grid,interpolation='none')
    line,=ax[1].plot(xrange, energies)
    # initialization function: plot the background of each frame
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=1000, interval=1)
