import numpy as np
import random
import math
import matplotlib.pyplot as plt

np.random.seed(10)
random.seed(10)

# Sidelength of the grid
GRID_SIZE = 4

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
def local_energy(grid, x_pos, y_pos):
    # dummy functie, vervang met je eigen functie
    return random.rand() * 3

def total_energy_simple(lattice):
    print("test")
    lattice_energy = []
    for y_pos, lattice_row in enumerate(lattice):
        lattice_row_energy = []
        for x_pos, s_i in enumerate(lattice_row):
            y_energy = lattice[(y_pos+1) % GRID_SIZE, x_pos % GRID_SIZE] + lattice[(y_pos-1) % GRID_SIZE, x_pos % GRID_SIZE]
            x_energy = lattice_row[(x_pos-1) % GRID_SIZE] + lattice_row[(x_pos+1) % GRID_SIZE]
            local_energy = (x_energy + y_energy)*s_i
            lattice_row_energy.append(local_energy)
        lattice_energy.append(lattice_row_energy)   
    print(lattice_energy) 
    return np.sum(lattice_energy)


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
    # print(str(left) + "\n" + str(right) + "\n" + str(below) + "\n" + str(up))
    return np.sum(energy * lattice)
#for y, row in enumerate(grid):
#...     for x, el in enumerate(row):
#...             print(el, grid[(y+1) % GRID_SIZE,x % GRID_SIZE])

def total_energy_simple(lattice):
    pass

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
        delta_energy = local_energy(grid, rand_x, rand_y)

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
        
        
        


