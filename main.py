# Import all nessesary libraries and setup environment
import numpy as np
import pandas as pd

materials = ['PP','PVDF','Mylar','Nylon']
Mylar_thickness = [0.9, 1.4, 2.0, 3.0, 3.5, 4.5, 6.0, 8.0, 10, 12.7, 4.0]
PVDF_thicknesses = [2,3,4,6]
# PE_thicknesses = [7] # TODO: Double check
PP_thicknesses = [2, 2.4, 3, 3.8, 4.8, 5.8, 6.8]
Nylon_thicknesses = [5] #TODO: Double check


mylar_nk = pd.read_csv('Refractive_Indices/Mylar_refractive_indices.csv')
nylon_nk = pd.read_csv('Refractive_Indices/Nylon_refractive_indices.csv')
pe_nk = pd.read_csv('Refractive_Indices/PE_refractive_indices.csv')
pp_nk = pd.read_csv('Refractive_Indices/PP_refractive_indices.csv')
pvdf_nk = pd.read_csv('Refractive_Indices/PVDF_refractive_indices.csv')


material_list = []
def add_material_thickness(material_name, thicknesses, nk_data):
    for thickness in thicknesses:
        material_list.append({
            'material': material_name,
            'thickness': thickness,
            'nk': nk_data
        })

# add_material_thickness('PE', PE_thicknesses, pe_nk)
add_material_thickness('PP', PP_thicknesses, pp_nk)
add_material_thickness('PVDF', PVDF_thicknesses, pvdf_nk)
add_material_thickness('Mylar', Mylar_thickness, mylar_nk)
add_material_thickness('Nylon', Nylon_thicknesses, nylon_nk)


import tmm

k_range = np.array(mylar_nk['Wavenumber'])
# Find indices where the wavelength is between 7 and 14 um
indices_in_range = np.where((10000/k_range >= 7) & (10000/k_range <= 14))
target_spectrum = np.zeros_like(k_range)
target_spectrum[indices_in_range] = 1

nonLWIR_penalty = 100

def construct_nd_list(individual, k):
    n_list = [1]  # Start with air
    d_list = [np.inf]  # Start with semi-infinite layer

    for i in individual:
        nk_data = material_list[i]['nk']
        thickness = material_list[i]['thickness']
        n_k_material = nk_data[nk_data['Wavenumber'] == k]
        # print(n_k_material)
        n_values = n_k_material['n_real'].values[0] + 1j * n_k_material['n_imag'].values[0]
        n_list.append(n_values)
        d_list.append(thickness)

    n_list.append(1)  # End with air
    d_list.append(np.inf)
    return n_list, d_list

def calc_spectrum(individual):

    absorption = []
    # Loop over wavenumbers
    for idx, k in enumerate(k_range):

        n_list, d_list = construct_nd_list(individual, k)
        wvl = 10000 / k  # Convert wavenumber to wavelength
        # Extract refractive indices at this wavenumber
        results = tmm.coh_tmm('s', n_list, d_list, 0, wvl)
        Absorption = 1 - results['R'] - results['T']
        absorption.append(Absorption)
    return np.array(absorption)

# TODO: Another approach will be to define a target absorption spectrum and then try to minimize the difference between the target and the actual absorption spectrum

def evaluate(individual):
    penalty = 0
    absorption = calc_spectrum(individual)
    mse = np.mean((absorption - target_spectrum) ** 2)
    return (mse,)

num_layers = 30

from deap import base, creator, tools
import random

# Define the problem as a minimization (since we're minimizing penalty)
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


def create_individual():
    individual = []
    for _ in range(num_layers):
        individual.append(random.choice(range(len(material_list))))
    return creator.Individual(individual)

def mutate_individual(individual):
    layer_idx = random.randrange(len(individual)) # Randomly select a layer to mutate
    new_index = random.choice(range(len(material_list)))
    while new_index == individual[layer_idx]:
        new_index = random.choice(range(len(material_list)))
    individual[layer_idx] = new_index
    return (individual,)

# Register the functions
toolbox.register('individual', create_individual)
toolbox.register('population', tools.initRepeat, list, toolbox.individual) # This is responsible for the initial guess (Can we replace certain populations with hand picked guess)
toolbox.register('evaluate', evaluate)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', mutate_individual)
toolbox.register('select', tools.selTournament, tournsize=3)

from deap import algorithms
random.seed(42)
population = toolbox.population(n=100)
num_generations = 1
cxpb = 0.5  # Crossover probability
mutpb = 0.2  # Mutation probability

# Statistics to keep track of the progress
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('min', np.min)
stats.register('avg', np.mean)
print("ok")
# Evolutionary algorithm
algorithms.eaSimple(population, toolbox, cxpb, mutpb, num_generations, stats=stats, verbose=True)

# Get the best individual
best_individual = tools.selBest(population, k=1)[0]
with open("best_individuals.txt", "w") as f:
    for ind in best_individual:
        f.write(f"Individu : {list(ind)}, Fitness : {ind.fitness.values[0]}\n")
print('Best individual:', best_individual)
print('Fitness:', best_individual.fitness.values[0])