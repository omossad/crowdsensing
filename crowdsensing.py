import numpy as np
import matplotlib.pyplot as plt
import math
import random
from deap import base
from deap import creator
from deap import tools
import time

## Constants
X = 10          # Model width
Y = 10          # Model Length
L = 40          # Number Of Locations
U = 120         # Maximum Number Of Users
actual_U = 120  # Actual Number Of Users

radius = 0.8    # Location Radius
B = 3           # Number of Bands
W = [1, 2, 3]   # Weights (Location-Band)
C = [1, 2, 3, 4, 5]  # Cost (User)
area = X * Y
x = np.random.rand(10000) * X
y = np.random.rand(10000) * Y
min_allowable_distance = radius * 1.5
keeperX = [x[0]]
keeperY = [y[0]]
Uxy = []
Lxy = [[x[0], y[0]]]
userX = random.sample(range(int(-radius * area), int(radius * area)), U / L)
userY = random.sample(range(int(-radius * area), int(radius * area)), U / L)
for u in range(U / L):
    Uxy.append([np.maximum(0, x[0] + userX[u] * radius / area), np.maximum(0, y[0] + userY[u] * radius / area)])
counter = 1
i = 1
while i < L:
    thisX = x[counter]
    thisY = y[counter]
    distances = []
    for j in range(len(Lxy)):
        distances.append(math.sqrt((thisX - keeperX[j]) ** 2 + (thisY - keeperY[j]) ** 2))
    min_distance = np.min(distances)
    if min_distance >= min_allowable_distance:
        keeperX.append(thisX)
        keeperY.append(thisY)
        Lxy.append([thisX, thisY])
        userX = random.sample(range(int(-radius * area), int(radius * area)), U / L)
        userY = random.sample(range(int(-radius * area), int(radius * area)), U / L)
        for u in range(U / L):
            Uxy.append(
                [np.maximum(0, thisX + userX[u] * radius / area), np.maximum(0, thisY + userY[u] * radius / area)])
        i = i + 1
    counter = counter + 1
chunks = []
num_chunks = 2 * math.pi / B
for i in range(B):
    chunks.append(i * num_chunks)
Lxy.append([radius, radius])
fig, ax = plt.subplots()
ax.set_xlim([0, X])  # set the bounds to be 1.5, 1.5
ax.set_ylim([0, Y])
circle = []

random_choice = np.random.randint(0,U,actual_U)
temp_Uxy = []
for i in range(len(random_choice)):
    temp_Uxy.append(Uxy[random_choice[i]])
Uxy = temp_Uxy
U = actual_U
for i in range(U):
    circle.append(plt.Circle((Uxy[i][0], Uxy[i][1]), 0.09, fill=True, color='r'))
for i in range(L):
    for c in range(B):
        endx = Lxy[i][0] + radius * math.cos(chunks[c])
        endy = Lxy[i][1] + radius * math.sin(chunks[c])
        ax.plot([Lxy[i][0], endx], [Lxy[i][1], endy], 'k')
    circle.append(plt.Circle((Lxy[i][0], Lxy[i][1]), radius, fill=False, color='k'))
for i in range(len(circle)):
    ax.add_artist(circle[i])
#plt.show()

location_band_weight = np.random.randint(low=W[0], high=W[-1] + 1, size=(L, B))
user_cost = np.random.randint(low=C[0], high=C[-1] + 1, size=U)

if B == 3:
    f = [0, 0.6, 0.85, 1]
elif B == 1:
    f = [0, 1]

user_sublocation = []
for i in range(L):
    for j in range(U):
        dx = Uxy[j][0] - Lxy[i][0]
        dy = Uxy[j][1] - Lxy[i][1]
        if math.hypot(dx, dy) <= radius:
            angle = math.atan2(dy, dx) % 2 * math.pi
            for k, e in reversed(list(enumerate(chunks))):
                if angle > e:
                    user_sublocation_entry = [j, i, k]
                    user_sublocation.append(user_sublocation_entry)
                    break

list_of_ind = np.zeros((L, U), dtype=np.int)
for i in range(L):
    for j in range(U):
        list_of_ind[i][j] = -1

location_users = []
intersect = [[0 for i in range(L)] for j in range(L)]
for i in range(L):
    users_list = []
    for j in range(len(user_sublocation)):
        if user_sublocation[j][1] == i:
            users_list.append(user_sublocation[j][0])
    location_users.append(users_list)
for i in range(len(location_users)):
    for j in range(len(location_users)):
        if i != j:
            if (bool(set(location_users[i]) & set(location_users[j]))):
                intersect[i][j] = 1
sublocation_users = []
for i in range(L):
    users_list = [[],[],[]]
    for j in range(len(user_sublocation)):
        if user_sublocation[j][1] == i:
            users_list[user_sublocation[j][2]].append(user_sublocation[j][0])
    sublocation_users.append(users_list)

overlap_percentage = np.divide(sum(x.count(1) for x in intersect), float(sum(len(y) for y in intersect)))
overlapIntensity = []
for i in range(len(intersect)):
    overlapIntensity.append(sum(intersect[i]))

total_cost = 0
for i in range(U):
    total_cost = total_cost + user_cost[i]
total_cost = 0.6 * total_cost
print "Maximum Cost = ", total_cost

def calculate_fitness(input_set):
    location_bands = []
    for i in range(L):
        temp = []
        temp.append(i)
        for b in range(B):
            temp.append(0)
        location_bands.append(temp)
    subloc = [[] for i in range(L)]
    for i in range(len(input_set)):
        for j in range(len(user_sublocation)):
            if i == user_sublocation[j][0]:
                loc = user_sublocation[j][1]
                if loc < L:
                    band = input_set[i]
                    if user_sublocation[j][2] not in subloc[loc]:
                        if band > 0:
                            location_bands[loc][band] = location_bands[loc][band] + 1
                            subloc[loc].append(user_sublocation[j][2])
    fitness = 0
    for i in range(len(location_bands)):
        loc = location_bands[i][0]
        for j in range(B):
            fitness = fitness + location_band_weight[loc][j] * f[location_bands[loc][j+1]]
    return fitness

def calculate_locations_fitness(input_set,island_users,used_locations):
    location_bands = []
    for i in used_locations:
        temp = []
        temp.append(i)
        for b in range(B):
            temp.append(0)
        location_bands.append(temp)
    subloc = [[] for i in range(len(used_locations))]
    for i in island_users:
        for j in range(len(user_sublocation)):
            if i == user_sublocation[j][0]:
                loc = user_sublocation[j][1]
                if loc < L:
                    band = input_set[island_users.index(i)]
                    if user_sublocation[j][2] not in subloc[used_locations.index(loc)]:
                        if band > 0:
                            location_bands[used_locations.index(loc)][band] = location_bands[used_locations.index(loc)][band] + 1
                            subloc[used_locations.index(loc)].append(user_sublocation[j][2])
    fitness = 0
    for i in range(len(location_bands)):
        loc = location_bands[i][0]
        for j in range(B):
            fitness = fitness + location_band_weight[loc][j] * f[location_bands[used_locations.index(loc)][j+1]]
    return fitness


def calculate_location_fitness(input_set,used_location):
    location_bands = []
    for b in range(B):
        location_bands.append(0)
    subloc = []
    for i in range(len(input_set)):
        for j in range(len(user_sublocation)):
            if location_users[used_location][i] == user_sublocation[j][0]:
                if user_sublocation[j][1] == used_location:
                    band = input_set[i]
                    if user_sublocation[j][2] not in subloc:
                        if band > 0:
                            location_bands[band-1] = location_bands[band-1] + 1
                            subloc.append(user_sublocation[j][2])
    fitness = 0
    for j in range(B):
        fitness = fitness + location_band_weight[used_location][j] * f[location_bands[j]]
    return fitness


def total_fitness(set):
    total_fitness = 0
    newSet = []
    for i in range(len(set)):
        newSet.append(set[i])
    newFitness = calculate_fitness(newSet)
    total_fitness = newFitness
    return total_fitness


def fitness_difference(set, entry):
    total_fitness = 0
    new_set = []
    for i in range(len(set)):
        new_set.append(set[i])
    old_fitness = calculate_fitness(new_set)
    new_set[entry[0]] = entry[1]
    new_fitness = calculate_fitness(new_set)
    total_fitness = new_fitness - old_fitness
    return total_fitness


def greedy():
    P = [0 for i in range(U)]
    users_added = [i for i in range(U)]
    calculated_cost = 0
    while len(users_added) > 0 and calculated_cost < total_cost:
        fitness_matrix = []
        for i in range(len(users_added)):
            currentUser = users_added[i]
            fitness_band_matrix = []
            for j in range(B):
                user_fitness = fitness_difference(P, [currentUser, j + 1])
                fitness_band_matrix.append(user_fitness)
            most_fit = fitness_band_matrix.index(max(fitness_band_matrix)) + 1
            cost_of_user = user_cost[currentUser]
            fitness_matrix.append([fitness_band_matrix[most_fit - 1] / cost_of_user, currentUser, most_fit])
        selectedUser = max(fitness_matrix)
        calculated_cost = calculated_cost + user_cost[selectedUser[1]]
        if calculated_cost > total_cost:
            calculated_cost = calculated_cost - user_cost[selectedUser[1]]
        else:
            P[selectedUser[1]] = selectedUser[2]
        users_added.remove(selectedUser[1])
    print "========================"
    print "GREEDY"
    print "========================"
    print "best individual = ", P
    print "cost ", calculated_cost
    return calculate_fitness(P)


def genetic_algorithm():
    print 'target utility = ',results_greedy
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("gene", random.randint, 0, B)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, U)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def maximize_fitness(individual):
        calculated_fitness = calculate_fitness(individual)
        cost_penalty = 0
        calculated_cost = 0
        for i in range(len(individual)):
            if individual[i] > 0:
                calculated_cost = calculated_cost + user_cost[i]
        if calculated_cost > total_cost:
            cost_penalty = -1*calculated_fitness
        return (calculated_fitness + cost_penalty),

    toolbox.register("evaluate", maximize_fitness)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=B, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=2)
    random.seed(0)
    pop = toolbox.population(n=60)
    for i in pop:
        temp_ind = generate_particle()
        for j in range(U):
            i[j] = temp_ind[j]
    CXPB, MUTPB, NGEN = 0.2, 0.3, 1500
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    best_fitness = []
    elitism = 0.05
    g = 0
    tolerance = 30
    flag = 0
    while g < NGEN:
    # for g in range(NGEN):
        offspring = toolbox.select(pop, int((1-elitism)*len(pop)))
        offspring.extend(tools.selBest(pop,int(elitism*len(pop))))
        random.shuffle(offspring)
        #offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        best_fitness.append(max(fits))
        if round(max(fits),2) >= round(results_greedy,2) and flag == 0:
            print "STOPPED AT", g
            flag = 1
            g = NGEN - tolerance
        g = g + 1
    best_ind = tools.selBest(pop, 1)[0]
    ga_cost = 0
    for i in range(len(best_ind)):
        if best_ind[i] > 0:
            ga_cost = ga_cost + user_cost[i]
    print "========================"
    print "Centralized GA"
    print "========================"
    print "best individual = ", best_ind
    print "cost ", ga_cost
    return calculate_fitness(best_ind)


def island_genetic_algorithm(island_users,used_locations,maximum_island_cost):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("gene", random.randint, 0, B)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, len(island_users))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def maximize_fitness(individual):
        calculated_fitness = calculate_locations_fitness(individual,island_users,used_locations)
        cost_penalty = 0
        calculated_cost = 0
        for i in range(len(individual)):
            if individual[i] > 0:
                calculated_cost = calculated_cost + user_cost[i]
        #if calculated_cost > maximum_island_cost:
        #    cost_penalty = -1*calculated_fitness
        cost_penalty = np.minimum(0, maximum_island_cost - calculated_cost)
        #cost_penalty = 0
        return (calculated_fitness + cost_penalty),
        #return (calculated_fitness / calculated_cost),

    toolbox.register("evaluate", maximize_fitness)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=B, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=10)
    random.seed(0)
    pop = toolbox.population(n=50)
    CXPB, MUTPB, NGEN = 0.2, 0.3, 200
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    for g in range(NGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
    best_ind = tools.selBest(pop, 1)[0]
    return best_ind


def permutate(location_number):
    user_no = location_users[location_number][0]
    fitnesses = []
    for i in range(B+1):
        #if i > 0:
        fitnesses.append(calculate_location_fitness([i], location_number)/user_cost[user_no])
        #else:
        #    fitnesses.append(0)
    best_ind = np.argmax(fitnesses)
    return [user_no,best_ind]

def permutate_locations(locations):
    user_no = location_users[locations[0]][0]
    fitnesses = []
    for i in range(B+1):
        #if i > 0:
        fitnesses.append(calculate_locations_fitness([i], locations))
        #else:
        #    fitnesses.append(0)
    best_ind = np.argmax(fitnesses)

    return [user_no,best_ind]

island_running_time = []
def island_GA():
    location_islands = []
    best_individual_IGA = np.zeros((U))
    for i in range(L):
        if(len(location_users[i]) == 1 and sum(intersect[i]) < 1):
            island_start_time = time.time()
            permutation_result = permutate(i)
            island_end_time = time.time()
            island_running_time.append(island_end_time - island_start_time)
            best_individual_IGA[permutation_result[0]] = permutation_result[1]
        elif(len(location_users[i]) > 0):
            location_islands.append(i)
    islands = []
    for i in location_islands:
        island = []
        island.append(i)
        for j in range(len(intersect[i])):
            if intersect[i][j] == 1:
                island.append(j)
        islands.append(island)
    final_islands = []
    while len(islands) > 0:
        first, rest = islands[0], islands[1:]
        first = set(first)
        lf = -1
        while len(first) > lf:
            lf = len(first)
            rest2 = []
            for r in rest:
                if len(first.intersection(set(r))) > 0:
                    first |= set(r)
                else:
                    rest2.append(r)
            rest = rest2

        final_islands.append(list(first))
        islands = rest

    calculated_cost = 0
    best_individual_IGA = [int(i) for i in best_individual_IGA]
    for i in range(len(best_individual_IGA)):
        if best_individual_IGA[i] > 0:
            calculated_cost = calculated_cost + user_cost[i]

    maximum_island_cost = (total_cost - calculated_cost) / len(final_islands)
    final_islands.sort(key=lambda x: len(x))
    print "ISLANDS", final_islands
    users_in_islands = []
    for i in range(len(final_islands)):
        island_users = []
        for j in range(len(final_islands[i])):
            #island_users.extend(location_users[final_islands[i][j]])
            in_first = set(island_users)
            in_second = set(location_users[final_islands[i][j]])
            in_second_but_not_in_first = in_second - in_first
            result = island_users + list(in_second_but_not_in_first)
            island_users = result
        users_in_islands.append(island_users)
        if len(island_users) == 1:
            island_start_time = time.time()
            permutation_result = permutate_locations(final_islands[i])
            island_end_time = time.time()
            island_running_time.append(island_end_time - island_start_time)
            best_individual_IGA[permutation_result[0]] = permutation_result[1]
            calculated_cost = 0
            for q in range(len(best_individual_IGA)):
                if best_individual_IGA[q] > 0:
                    calculated_cost = calculated_cost + user_cost[q]
            if i < len(final_islands)-1:
                maximum_island_cost = (total_cost - calculated_cost) / (len(final_islands)-i-1)
        else:
            island_start_time = time.time()
            best_individual_in_island = island_genetic_algorithm(island_users,final_islands[i],maximum_island_cost)
            island_end_time = time.time()
            island_running_time.append(island_end_time - island_start_time)
            for ind in range(len(best_individual_in_island)):
                best_individual_IGA[island_users[ind]] = best_individual_in_island[ind]
            calculated_cost = 0
            for q in range(len(best_individual_IGA)):
                if best_individual_IGA[q] > 0:
                    calculated_cost = calculated_cost + user_cost[q]
            if i < len(final_islands)-1:
                maximum_island_cost = (total_cost - calculated_cost) / (len(final_islands)-i-1)

    print "ISLANDS USERS", users_in_islands
    best_individual_IGA = [int(i) for i in best_individual_IGA]
    calculated_cost = 0
    for i in range(len(best_individual_IGA)):
        if best_individual_IGA[i] > 0:
            calculated_cost = calculated_cost + user_cost[i]
    print "========================"
    print "Island GA"
    print "========================"
    print "best individual = ", best_individual_IGA
    print "cost ", calculated_cost
    return calculate_fitness(best_individual_IGA)

def generate_particle():
    particle = [0 for i in range(U)]
    calculated_cost = 99999
    while calculated_cost > total_cost:
        calculated_cost = 0
        for i in range(len(sublocation_users)):
            for j in range(B):
                if len(sublocation_users[i][j]) > 0:
                    particle[random.choice(sublocation_users[i][j])] = random.randint(1,B)
        for i in range(len(particle)):
            if particle[i] > 0:
                calculated_cost = calculated_cost + user_cost[i]
        while calculated_cost > total_cost:
            random_num = random.randint(0, U - 1)
            particle[random_num] = 0
            calculated_cost = 0
            for i in range(U):
                if particle[i] > 0:
                    calculated_cost = calculated_cost + user_cost[i]
    return particle


def randomize_particle():
    particle = [0 for i in range(U)]
    calculated_cost = 99999
    while calculated_cost > total_cost:
        calculated_cost = 0
        particle = np.random.randint(B+1, size=U)
        for i in range(len(particle)):
            if particle[i] > 0:
                calculated_cost = calculated_cost + user_cost[i]
        while calculated_cost > total_cost:
            random_num = random.randint(0, U - 1)
            particle[random_num] = 0
            calculated_cost = 0
            for i in range(U):
                if particle[i] > 0:
                    calculated_cost = calculated_cost + user_cost[i]
    return particle

def merge_particle(part,part_best,glob_best):
    new_part = [0 for u in range(U)]
    for i in range(U):
        a = part[0][i]
        b = part_best[0][i]
        c = glob_best[0][i]
        d = 0
        if a + b == 0:
            d = c
        elif a + c == 0:
            d = b
        elif b + c == 0:
            d = a
        else:
            user_loc_subloc = []
            a_weight = b_weight = c_weight = 0
            for j in range(len(user_sublocation)):
                if user_sublocation[j][0] == i:
                    loc = user_sublocation[j][1]
                    subloc = user_sublocation[j][2]
                    user_loc_subloc.append([loc,subloc])

                    for j in range(len(sublocation_users[loc][subloc])):
                        if new_part[sublocation_users[loc][subloc][j]] == a:
                            a_weight = a_weight - 5
                        if new_part[sublocation_users[loc][subloc][j]] == b:
                            b_weight = b_weight - 5
                        if new_part[sublocation_users[loc][subloc][j]] == c:
                            c_weight = c_weight - 5

                    a_weight = a_weight + location_band_weight[loc][a-1]
                    b_weight = b_weight + location_band_weight[loc][b-1]
                    c_weight = c_weight + location_band_weight[loc][c-1]
            #a_weight = fitness_difference(new_part,[i,a])
            #b_weight = fitness_difference(new_part,[i,b])
            #c_weight = fitness_difference(new_part,[i,c])
            if a_weight > b_weight:
                if a_weight > c_weight:
                    d = a
                else:
                    d = c
            else:
                if b_weight > c_weight:
                    d = b
                else:
                    d = c
        new_part[i] = d
    calculated_cost = 0
    for i in range(U):
        if new_part[i] > 0:
            calculated_cost = calculated_cost + user_cost[i]

    while calculated_cost > total_cost:
        random_num = random.randint(0, U-1)
        new_part[random_num] = 0
        calculated_cost = 0
        for i in range(U):
            if new_part[i] > 0:
                calculated_cost = calculated_cost + user_cost[i]
    return [new_part,calculate_fitness(new_part)]

def pso():
    num_particles = 300
    particles = []
    particles_fitnesses = []
    particles_best = []
    for i in range(num_particles):
        particles.append([])
        if i < int(num_particles * 0.8):
            new_particle = generate_particle()
        else:
            new_particle = randomize_particle()
        particles[i].append(new_particle)
        particles[i].append(calculate_fitness(new_particle))
        particles_fitnesses.append(particles[i][1])
    global_best_particle_index = np.argmax(particles_fitnesses)
    global_best_particle = particles[global_best_particle_index]
    particles_best = particles
    lifetime = 70
    for q in range(lifetime):
        particles_fitnesses = []
        for i in range(num_particles):
            new_particle = merge_particle(particles[i],particles_best[i],global_best_particle)
            particles[i] = new_particle
            if new_particle[1] > particles_best[i][1]:
                particles_best[i] = new_particle
            particles_fitnesses.append(new_particle[1])
        new_global_best_particle_index = np.argmax(particles_fitnesses)
        new_global_best_particle = particles[new_global_best_particle_index]
        if new_global_best_particle[1] > global_best_particle[1]:
            global_best_particle = new_global_best_particle
    calculated_cost = 0
    for i in range(len(global_best_particle[0])):
        if global_best_particle[0][i] > 0:
            calculated_cost = calculated_cost + user_cost[i]
    print "pso cost = ", calculated_cost
    return global_best_particle


def main():
    print "========================"
    print "PARAMETERS"
    print "========================"
    print "# of USERS: ",U
    print "# of LOCATIONS: ",L
    print "Uxy =", Uxy
    print "Lxy =",Lxy
    print "user_cost =",list(user_cost)
    print "location_band_weight =",np.array2string(location_band_weight, separator=', ')
    print "========================"

    global results_greedy
    start_time_greedy = time.time()
    results_greedy = greedy()
    end_time_greedy = time.time()
    time_greedy = end_time_greedy - start_time_greedy

    start_time_island_GA = time.time()
    results_island_GA = island_GA()
    end_time_island_GA = time.time()
    time_island_GA = end_time_island_GA - start_time_island_GA
    time_island_GA = time_island_GA + np.max(island_running_time)
    time_island_GA = time_island_GA - np.sum(island_running_time)

    start_time_centralized_GA = time.time()
    results_centralized_GA = genetic_algorithm()
    end_time_centralized_GA = time.time()
    time_centralized_GA = end_time_centralized_GA - start_time_centralized_GA

    start_time_pso = time.time()
    results_pso = pso()
    end_time_pso = time.time()
    time_pso = end_time_pso - start_time_pso

    print "========================"
    print "RESULTS"
    print "========================"
    print "Centralized GA UTILITY VALUES: ",results_centralized_GA
    print "Centralized GA TIME: ",time_centralized_GA
    print "Island GA UTILITY VALUES: ",results_island_GA
    print "Island GA TIME: ",time_island_GA
    print "GREEDY UTILITY VALUES: ",results_greedy
    print "GREEDY TIME: ",time_greedy
    print "PSO UTILITY VALUES: ", results_pso[1]
    print "PSO TIME: ", time_pso
    print "========================"
    print "Utilities"
    print "========================"
    print results_centralized_GA
    print results_island_GA
    print results_greedy
    print results_pso[1]
    print "========================"
    print "Time"
    print "========================"
    print time_centralized_GA
    print time_island_GA
    print time_greedy
    print time_pso


main()
