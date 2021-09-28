#https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
# %%
from numpy.random import randint
from numpy.random import rand
import pandas as pd

# %%
# objective function
def objective(x):
	return x[0]**2.0 + x[1]**2.0
 
# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		# extract the substring
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		# convert bitstring to a string of chars
		chars = ''.join([str(s) for s in substring])
		# convert string to integer
		integer = int(chars, 2)
		# scale integer to desired range
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		# store
		decoded.append(value)
	return decoded
 
# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]
 
# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]
 
# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]
# genetic algorithm
def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
	# initial population of random bitstring
	pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
	# keep track of best solution
	best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))
	# enumerate generations
	for gen in range(n_iter):
		# decode population
		decoded = [decode(bounds, n_bits, p) for p in pop]
		# evaluate all candidates in the population
		scores = [objective(d) for d in decoded]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				#print(">%d, new best f(%s) = %f" % (gen,  decoded[i], scores[i]))
		# select parents
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
	return [best, best_eval]
 
# define range for input
bounds = [[0, 1], [0, 11]]
# define the total iterations
n_iter = 100
# bits per variable
n_bits = 16
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / (float(n_bits) * len(bounds))
# %%
r_mut
# %%

decode(bounds, n_bits, bitstring)
#%%


# perform the genetic algorithm search
best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
decoded = decode(bounds, n_bits, best)
print('f(%s) = %f' % (decoded, score))
# %%


cost = pd.read_csv('cost.csv', index_col=0)
demand = pd.read_csv('demand.csv', index_col=0)
prepaid = pd.read_csv('prepaid.csv', index_col=0)
# set
T = list(range(len(cost.index)))
I = list(cost.columns)
J = ['A', 'B', 'C']
cit = {}
pit = {}
dit = {}
mijt = {}
J_bound = {'A': 2, 'B':3, 'C': 5}
# Parameters
for i, item in enumerate(cost.columns):
    for t, date in enumerate(cost.index):
        cit[item, t] = int(cost.iloc[t, i])

for i, item in enumerate(prepaid.columns):
    for t, date in enumerate(prepaid.index):
        pit[item, t] = int(prepaid.iloc[t, i])

for i, item in enumerate(demand.columns):
    for t, date in enumerate(demand.index):
        dit[item, t] = int(demand.iloc[t, i])

for i, item in enumerate(demand.columns):
    for j  in J :
        for t, date in enumerate(demand.index):
            mijt[item,j, t] = 1
#%%
cit

# %%
pit

# %%
dit
# %%
mijt
#%%
J_bound
#%%
x_ub = 11
x_lb = 0

def generation_xijt():
    xijt = {}
    for i in I:
        for j in J:
            for t  in T :
                xijt[i, j, t] = randint(x_lb, x_ub)
    return xijt

def dict2bitstring(xijt):
	return list(xijt.values())

def bitstring2dict(bitstring):
	for idx, value in enumerate(bitstring):
		xijt[xijt_keys[idx]] = value
	return xijt

def objective(xijt):
	uit = {}
	for i in I:
		for t in T:
			u = dit[i, t] - sum(xijt[i, j, t] for j in J)
			if u >=0 :
				uit[i, t] = u
			else:
				uit[i, t] = 0

	objective = sum(uit[i, t]*cit[i, t]*pit[i, t] for i in I for t in T)
	return objective


xijt = generation_xijt()
xijt_keys = list(xijt.keys())

bitstring = dict2bitstring(xijt)
bitstring2dict(bitstring)
objective(xijt)


# %%

def constraint_check(xijt):
    #  Constraint 2
    check_list = []
    for i in I:
        for t in T:
            check = sum(mijt[i, j, t]*xijt[i, j, t] for i in I) <=  600
            check_list.append(check)
    const_2 = all(i==True for i in check_list)
    return (const_2 == True) #& (const_1 == True)
# %%

def generation():
	### objective function
	xijt = generation_xijt()
	uit = generation_uit()
	n = 0
	while n < n_pop:
		for i in range(150000):
			if constraint_check(uit, xijt) == True:
				#print(f'find!, uit: {uit}, xijt: {xijt}')

				n += 1
				return uit, xijt
			else:
				raise ValueError('No feasible solution')

uit, xijt = generation()
# %%
xijt = generation_xijt()
uit = {}
for i in I:
	for t in T:
		u = dit[i, t] - sum(xijt[i, j, t] for j in J)
		if u >=0 :
			uit[i, t] = u
		else:
			uit[i, t] = 0

objective = sum(uit[i, t]*cit[i, t]*pit[i, t] for i in I for t in T)
        
print('Done!', objective) 
# %%
uit
#%%
oracle = pd.read_csv('solution.csv')
oracle[oracle['variable'].str.contains('x_a')]
# %%
dit['a', 1]
# %%
dit
# %%
