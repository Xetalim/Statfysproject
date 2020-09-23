import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib import animation

import time
import datetime as dt

import os

np.random.seed(1)
random.seed(1)

# Sidelength of the grid
# GRID_SIZE = 100
# decide whether or not to animate the simulation
ANIMATE_SIM = False
# Generating the grid, with dtype integer
# grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int) + 1
# grid = (np.random.randint(0, 2, size=(GRID_SIZE, GRID_SIZE)) * 2) - 1

def accept_change(delta_energy, t_red):
    # calculate the chance given a t_red and change in energy
    p = math.exp(- delta_energy / t_red)
    # generate a random number between 0 and 1. This function is quicker
    # than np.random.rand() by a factor 100.
    num = random.random()
    # equivalent to if p> num: return True, if p <= num: return False
    return p > num


def local_energy_flipped(lattice, x_pos, y_pos, GRID_SIZE):
    # doesn't have a minus sign because lattice[x,y] has been flipped
    # get the left, right, top and bottom spin value
    left = lattice[y_pos, (x_pos - 1) % GRID_SIZE]
    right = lattice[y_pos, (x_pos + 1) % GRID_SIZE]
    top = lattice[(y_pos + 1) % GRID_SIZE, x_pos]
    bottom = lattice[(y_pos - 1) % GRID_SIZE, x_pos]
    # we multiply the spin values with the spin value at x_pos, y_pos
    # factor 2, because we don't want to count anything double
    delta_energy = lattice[y_pos, x_pos] * 2 * (left + right + top + bottom)
    return delta_energy

def total_energy_simple(lattice, GRID_SIZE):
    lattice_energy = []
    # comments!!!
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
    

def total_energy_quick(lattice, GRID_SIZE):
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
    return 0.5 * np.sum(energy * lattice, dtype=np.int64)


# def calc_stop_sim(perc, energy_list, i, index, calc_freq):
#     a = np.average(energy_list[max(i // calc_freq - 50 * (i // index) // calc_freq,0):max(i // calc_freq - 25 * (i // index) // calc_freq,1)])
#     b = np.average(energy_list[max(i // calc_freq - 25 * (i // index) // calc_freq,0):i // calc_freq])

#     newperc = np.abs((a-b)/(a) * 100)
#     min_i = max(i // calc_freq - 25 * (i // index) // calc_freq,0)
#     max_i = i // calc_freq
#     # print(a,b)
#     # print("{} out of {}, diff = {:.2f}".format(index, 1000, newperc))

    
#     stop = (newperc < 2 and perc < 2)
#     return stop, newperc, (min_i, max_i)

def simulate(t_red, grid, iterations, calc_freq):
    # initialise equilibrium checkpoints
    # these will be 10k, 30K, 50K, 70K, 100K, 300K, 500K and so forth
    a = [1,3,5,7]
    b = 10**np.arange(4, 8)
    checkpoints = (b[:,np.newaxis]*a).flatten()
    
    # retrieve size of grid
    GRID_SIZE = len(grid[:,0])
    
    # initialise energy and magnetisation, we will update these values
    energy = total_energy_quick(grid, GRID_SIZE)
    mag = np.sum(grid)
    
    # initialise energy and magnetisation lists, to append new values to
    energy_list = []
    mag_list = []
    
    
    
    # initialise the percentage difference between a range of iterations
    # 
    old_percentage = 100
    
    for i in range(0, iterations):
        # initialise accept value
        accept_flip = False
        # choosing random coordinates
        rand_x = int(random.random() * GRID_SIZE)
        rand_y = int(random.random() * GRID_SIZE)
        # is the same as
        # rand_x = np.random.randint(0, GRID_SIZE)
        # but about 1000 times as fast
        
        # calculate the difference in energy
        delta_energy = local_energy_flipped(grid, rand_x, rand_y, GRID_SIZE)

        # if delta_energy is less than zero, accept the change
        if delta_energy <= 0:
            accept_flip = True
            
        # if delta_energy is more than zero, accept the change
        # with the probability as given in the algorithm
        elif accept_change(delta_energy, t_red):
            accept_flip = True
        # if we accept a flip, we flip the spin and update the energy and
        # magnetisation
        if accept_flip:
            grid[rand_y, rand_x] *= -1
            energy = energy + delta_energy
            mag = mag + 2 * grid[rand_y, rand_x]
        
        # calculate and append all observables, every calc_freq times.
        # this means that our energy_list is calc_freq smaller than the 
        # amount of iterations.
        if i % calc_freq == 0:
            
            # append our magnetisation and energy values to their lists
            energy_list.append(energy)
            mag_list.append(mag)

            # in checkpoints, there is a list of iteration vallues for which
            # we check if we are in equilibrium.
            # If we are in equilibrium, we will stop the simulation
            
            # initialise stop value
            stop = False
            
            # check if we are at a checkpoint, and if we are, get the 
            # location of our value in the checkpoints list
            if i in checkpoints:
                index = np.where(checkpoints == i)[0][0]
                
                # we are going to use index - 2, so make sure we can actually
                # subtract 2.
                if index > 2:
                    
                    # we create 2 ranges, between min_1 and max_1, and
                    # between min_2 and max_2. We divide by calc_freq to
                    # keep our indices in line with the smaller size of the
                    # energy list.
                    # this would for example be between 10k and 30k, and
                    # between 30k and 50k
                    min_1 = checkpoints[max(index - 2,0)] // calc_freq
                    max_1 = checkpoints[max(index-1,0)] // calc_freq
                    min_2 = checkpoints[max(index-1,0)] // calc_freq
                    max_2 = checkpoints[index] // calc_freq
                    
                    # the big range, consisting of both smaller ranges
                    equilibrium_range = (min(min_1,max_1),max(min_2,max_2))
                     
                    # calculate the average energies over these ranges
                    average_1 = np.average(energy_list[min(min_1,max_1): max(min_1,max_1)])
                    average_2 = np.average(energy_list[min(min_2,max_2): max(min_2,max_2)])

                    # calculate the prercentagewise difference between
                    # these energies.
                    new_percentage = abs((average_1 - average_2) / average_1 * 100)
                   
                    # if the difference between the two averages is small,
                    # and this was also the case for the previous check,
                    # we stop the simulation.
                    # this means that over our whole range, our average
                    # has not changed much, so we are in equilibrium.
                    if new_percentage < 1.6 and old_percentage < 1.6:
                        stop = True
                        print("tred: {} indices: {}".format(t_red, equilibrium_range))
                        break
                    # update the old_percentage, so we can use it again
                    # next time.
                    old_percentage = new_percentage
                    print("tred: {:.2f} i: {:.4e}, perc: {:.2f}, index: {:.4e}".format(t_red, i, new_percentage, index))
           # we want to stop the simulation because we are in equilibrium,
           # so we break the iterations loop.
            if stop:
                break

    output_data = {"energy":energy_list, "magnetisation":mag_list, "grid":grid}
    return output_data, equilibrium_range

def calculate_values(en, maglist, indices):
    capacity = np.var(en[indices[0]:indices[1]])
    mag = np.average(maglist[indices[0]:indices[1]])
    mag_abs = np.average(np.abs(maglist[indices[0]:indices[1]]))
    avg_en = np.average(en[indices[0]:indices[1]])
    return capacity, mag, mag_abs, avg_en

def calc_remaining_time(starttime, cur_iter, max_iter):

    telapsed = time.time() - starttime
    testimated = (telapsed/cur_iter)*(max_iter)

    finishtime = starttime + testimated
    finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time

    lefttime = testimated-telapsed  # in seconds

    return (int(telapsed), int(lefttime), finishtime)

def multiprocessing_wrapper(GRID_SIZE, temp_reduced, max_iterations, calc_freq, random):
    if random:
        grid = (np.random.randint(0, 2, size=(GRID_SIZE, GRID_SIZE),dtype=int) * 2) - 1
    else:
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int) + 1
    data, equilibrium_range = simulate(temp_reduced,
                                       grid,
                                       max_iterations,
                                       calc_freq)
    
    capacity, mag, mag_abs, avg_en = calculate_values(data["energy"],
                                                      data["magnetisation"],
                                                      equilibrium_range)

    return (temp_reduced, capacity, mag, mag_abs, avg_en)
def savearray(relative_path, name, array):
    if not os.path.exists(relative_path):
        os.makedirs(relative_path)
    with open(relative_path + "/" + name, "wb") as file:
        np.save(file, array)
    

def sim_multiple(GRID_SIZE,
                 temp_steps,
                 min_temp,
                 max_temp,
                 max_iterations,
                 calc_freq,
                 averages,
                 random=False,
                 multiprocessing=False):
    
    tredlist = np.linspace(min_temp, max_temp, temp_steps)
    clist = []
    maglist = []
    mag_abslist = []
    enlist = []
    
    resultarr = np.empty((averages, temp_steps, 5))
    
    if multiprocessing:
        import multiprocessing
        from joblib import Parallel, delayed
        num_cores = multiprocessing.cpu_count()

        for j in range(0, averages):
            results = np.array(Parallel(
                n_jobs=num_cores, verbose=50)(
                    delayed(multiprocessing_wrapper)(
                        GRID_SIZE, 
                        tred, 
                        max_iterations, 
                        calc_freq, 
                        random) for tred in tredlist))
            resultarr[j, :, :] = results
        resultarr = resultarr
    
    if not multiprocessing:
        for avg_index in range(0, averages):
            start = time.time()
            for j, tred in enumerate(tredlist):
                # maybe use the same grid every time?
                if random:
                    grid = (np.random.randint(0, 2, size=(GRID_SIZE, GRID_SIZE)) * 2) - 1
                else:
                    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int) + 1
                data, equilibrium_range = simulate(tred,
                                                   grid,
                                                   max_iterations,
                                                   calc_freq)
                
                capacity, mag, mag_abs, avg_en = calculate_values(
                                                      data["energy"],
                                                      data["magnetisation"],
                                                      equilibrium_range)
                clist.append(capacity)
                maglist.append(mag)
                mag_abslist.append(mag_abs)
                enlist.append(avg_en)
                
                if j % 5 == 0:
                    time_left = calc_remaining_time(start, j+1, temp_steps)
                    print("time elapsed: %s(s), time left: %s(s), estimated finish time: %s"%time_left)
            
            results = np.array((tredlist, clist, maglist, mag_abslist, enlist))
            resultarr[avg_index, :, :] = results
    namestring = "data_{}_{}_{:.1f}-{:.1f}_{}_{}.dat".format(
                                                 GRID_SIZE,
                                                 temp_steps,
                                                 min_temp,
                                                 max_temp,
                                                 calc_freq,
                                                 averages)
    savearray("data",namestring,resultarr)
    # resultarr[:,:,0] is the temp_reduced list for average 1
    # resultarr[:,:,1] is heat capacity, 2 is magnetisation, 3 is absolute magnetisation, 4 is energy
    # resultarr[1,:,0] gives the temp_reduced for average 2 and so forth
    return resultarr



# plotting the heat capacity etc. vs. the reduced temperature
# we give our arguments
GRID_SIZE = 20
temp_steps = 20
min_temp = 1
max_temp = 4
max_iterations = 10**7
calc_freq = 10
averages = 1

# the function with our arguments
values_sim_multiple = sim_multiple(GRID_SIZE,
                 temp_steps,
                 min_temp,
                 max_temp,
                 max_iterations,
                 calc_freq,
                 averages,
                 random=False,
                 multiprocessing=True)
t_red = values_sim_multiple[0,:,0]
heat_capacity = values_sim_multiple[0,:,1]
heat_capacity_average = np.average(values_sim_multiple[:,:,1],axis=0)
heat_capacity_stdev = np.std(values_sim_multiple[:,:,1],axis=0)

magnetisation = values_sim_multiple[0,:,2]
magnetisation_average = np.average(values_sim_multiple[:,:,2],axis=0)
magnetisation_stdev = np.std(values_sim_multiple[:,:,2],axis=0)

absolute_magnetisation = values_sim_multiple[0,:,3]
absolute_magnetisation_average = np.average(values_sim_multiple[:,:,3],axis=0)
absolute_magnetisation_stdev = np.std(values_sim_multiple[:,:,3],axis=0)

energy = values_sim_multiple[0,:,4]
energy_average = np.average(values_sim_multiple[:,:,4],axis=0)
energy_stdev = np.std(values_sim_multiple[:,:,4],axis=0)


fig,ax = plt.subplots(4, sharex=True, sharey=False, figsize=(5, 15))

if averages == 1:
    ax[0].errorbar(t_red,heat_capacity,fmt='b.')
    ax[1].errorbar(t_red,magnetisation,fmt='r.')
    ax[2].errorbar(t_red,absolute_magnetisation,fmt='k.')
    ax[3].errorbar(t_red,energy,fmt='m.')
else:
    ax[0].errorbar(t_red,heat_capacity_average, yerr=heat_capacity_stdev, fmt='b.')
    ax[1].errorbar(t_red,magnetisation_average, yerr=magnetisation_stdev, fmt='r.')
    ax[2].errorbar(t_red,absolute_magnetisation_average, yerr=absolute_magnetisation_stdev, fmt='k.')
    ax[3].errorbar(t_red,energy_average, yerr=energy_stdev,fmt='m.')

ax[0].set_ylabel(r'$\langle C \rangle$') # fix units
ax[1].set_ylabel(r'$\langle m \rangle$') # fix units
ax[2].set_ylabel(r'$\langle |m| \rangle$') # fix units
ax[3].set_ylabel(r'$\langle E \rangle$') # fix units

fig.savefig("data/data_{}_{}_{}-{}_{}_{}.png".format(
    GRID_SIZE, temp_steps, min_temp, max_temp, calc_freq, averages))

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
    t_red = 5
    
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
