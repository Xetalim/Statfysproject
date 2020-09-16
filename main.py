import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import Animation

np.random.seed(10)
random.seed(10)

# Sidelength of the grid
GRID_SIZE = 1000

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
    # doesnt have a minus sign because lattice[x,y] has been flipped
    left = lattice[y_pos, x_pos] * lattice[y_pos, (x_pos - 1) % GRID_SIZE]
    right = lattice[y_pos, x_pos] * lattice[y_pos, (x_pos + 1) % GRID_SIZE]
    top = lattice[y_pos, x_pos] * lattice[(y_pos + 1) % GRID_SIZE, x_pos]
    bottom = lattice[y_pos, x_pos] * lattice[(y_pos - 1) % GRID_SIZE, x_pos]
    # factor 2, because we don't want to count anything double
    delta_energy = 2 * (left + right + top + bottom)
    return delta_energy

def total_energy_simple(lattice):
    # print("test")
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

def simulate(t_red, grid, iterations):
    energy = total_energy_quick()
    for i in range(0, iterations):
        accept = False
        # choosing random coordinates
        rand_x = int(random.random() * GRID_SIZE)
        rand_y = int(random.random() * GRID_SIZE)
        # is the same as
        # rand_x = np.random.randint(0, GRID_SIZE)
        # but about 100 times as fast

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
        



