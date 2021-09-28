#%%
from mip import Model, xsum, minimize, BINARY, INTEGER

import pandas as pd
# sample2 엑셀 데이터 불러오기
df_sample = pd.read_csv('sample2.csv')

cost = df_sample[['item', 'T', 'C']].drop_duplicates().dropna(axis=0)
demand = df_sample[['item', 'T', 'D']].drop_duplicates().dropna(axis=0)
prepaid = df_sample[['item', 'T', 'P']].drop_duplicates().dropna(axis=0)
cycle = df_sample[['item', 'J', 'T', 'M']].drop_duplicates().dropna(axis=0)

ijt_df = pd.DataFrame(df_sample[['item', 'J', 'T']].value_counts()).reset_index()[['item', 'J', 'T']]
ijt_df = ijt_df.sort_values(['item']).reset_index(drop=True)
ijt_set = set()
for index, info in ijt_df.iterrows():
    ijt_set.add((info['item'], info['J'], info['T']))
ijt_set
# %%
# variables infomation
T = list(set(df_sample['T']))
I = list(set(df_sample['item']))
J = list(set(df_sample['J']))

# Parameters
cit = {}
pit = {}
dit = {}
mijt = {}

# cit - 단가
for index, info in cost.iterrows():
    cit[info['item'], info['T']] = info['C']
# pit - 선급품/일반품
for index, info in prepaid.iterrows():
    pit[info['item'], info['T']] = info['P']
# dit - 생산목표량
for index, info in demand.iterrows():
    dit[info['item'], info['T']] = info['D']
# mijt - cycle time
for index, info in cycle.iterrows():
    mijt[info['item'], info['J'], info['T']] = info['M']

# adding missing values
for i in I:
    for t in T:
        key = (i, t)
        if key not in cit:
            cit[i, t] = 0
for i in I:
    for t in T:
        key = (i, t)
        if key not in pit:
            pit[i, t] = 0
for i in I:
    for t in T:
        key = (i, t)
        if key not in dit:
            dit[i, t] = 0
for i in I:
    for j in J:
        for t in T:
            key = (i, j, t)
            if key not in mijt:
                mijt[i, j, t] = 0

# variables
m = Model("Unist_optimization")
xijt = {
    (i, j, t): m.add_var(var_type=INTEGER, name="x_%s,%s,%s" % (i, j, t))
    for i in I for j in J for t in T
}
uit = {
    (i, t): m.add_var(var_type=INTEGER, name="u_%s,%s" % (i, t))
    for i in I for t in T
}

# objective
m.objective = minimize(
    xsum(uit[i, t]*cit[i, t]*pit[i, t] for i in I for t in T)
    )

# constraint
# Constraint 1:-
for i in I:
    for t in T:
        m += (uit[i, t] + xsum(xijt[i, j, t] for j in J) == dit[i, t])
# Constraint 2:-
for j in J:
    for t in T:
        m += (xsum(mijt[i, j, t]*xijt[i, j, t] for i in I) <= 600)
# Constraint 3, 4:-
for i in I:
    for j in J:
        for t in T:
            m += (xijt[i, j, t] >= 0)
for i in I:
    for t in T:
        m += (uit[i, t] >= 0)
# Constraint 5
all_set = set()
for i in I:
    for j in J:
        for t in T:
            if (i, j, t) not in ijt_set:
                m += (xijt[i, j, t] == 0)

m.write('model.lp')
m.read('model.lp')
m.optimize()
m.objective_value

solution = []
for i in I:
    for j in J:
        for t in T:
            solution.append([xijt[i, j, t].name, xijt[i, j, t].x])
            solution.append([uit[i, t].name, uit[i, t].x])
solution = pd.DataFrame(
    solution,
    columns=['variable', 'solution'])
solution.sort_values(['variable', 'solution'], inplace=True)
solution.to_csv('solution.csv', index=False)

solution = pd.read_csv('solution.csv')
solution[solution['solution'] > 0]
# %%
################################################################################
# Genetic Alogorithm
################################################################################
from numpy.random import randint
from numpy.random import rand
import pandas as pd
import math

# %%
df_raw = pd.read_excel('생산미결리스트 등 샘플데이터.xlsx', sheet_name=None)
machine_bound_info = df_raw['CYCLETIME'][1:].reset_index(drop=True)
item_info = df_raw['생산미결리스트(for cost)'].reset_index(drop=True)
# item-단가 unique 하지 않음 (수량이 다르므로) ==> item 별로 합산을 해서 총 수량을 하기엔... 영업납기일이 다름..! 흠..
# 데이터 자체를 쓰기가.. 흠..demand 부터 뽑는것이 애매함
item_info[['중산도면', '수량']].value_counts()
#item_info[item_info['중산도면']=='057386']

machine_bound_info[['JSDWG', 'MCNO']].value_counts() # 이건 unique함!

machine_bound = {}
for index, info in machine_bound_info.iterrows():
    item = info['JSDWG']
    machine = info['MCNO']
    machine_bound[item, machine] = math.ceil(info['10h'])
item_cost = {}
for index, info in item_info.iterrows():
    item = info['중산도면']
    item_cost[item] = int(info['단가'])
item_demand = {}
for index, info in item_info.iterrows():
    item = info['중산도면']
    item_demand[item] = int(info['수량'])
item_urgent = {}
for index, info in item_info.iterrows():
    item = info['중산도면']
    if info['선급'] != info['선급']:
        item_urgent[item] = 1
    else:
        item_urgent[item] = 0
item_urgent


# %%
mijt
# %%


def generation_xijt():
    xijt = {}
    for i in I:
        for j in J:
            for t in T:
                if mijt[i, j, t] == 0:
                    xijt[i, j, t] = 0
                else:
                    xijt[i, j, t] = randint(0, mijt[i, j, t])
    return xijt


def decode(mijt, xijt):
    for i in xijt:
        ub = mijt[i]
        if xijt[i] > ub:
            if ub == 0:
                xijt[i, j, t] = 0
            else:
                xijt[i, j, t] = randint(0, mijt[i, j, t])
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
            if u >= 0:
                uit[i, t] = u
            else:
                uit[i, t] = 0
    objective = sum(uit[i, t]*cit[i, t]*pit[i, t] for i in I for t in T)
    return objective


# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def crossover(p1, p2, r_cross):
    p1 = dict2bitstring(p1)
    p2 = dict2bitstring(p2)
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
# %%
xijt = generation_xijt()
xijt_keys = list(xijt.keys())
bitstring = dict2bitstring(xijt)
bitstring
xijt = bitstring2dict(bitstring)
xijt
# %%

n_iter = 10
# bits per variable
n_bits = 16
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / (float(n_bits) * len(xijt))

# %%
def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    pop = [generation_xijt() for _ in range(n_pop)]
    best, best_eval = 0, objective(decode(mijt, pop[0]))

    for gen in range(n_iter):    
        decoded = [decode(mijt, p) for p in pop]
        # evaluate all candidates in the population
        scores = [objective(d) for d in decoded]

        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]

        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]

        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(bitstring2dict(c))
        pop = children
    return [best, best_eval]

# %%
best, score = genetic_algorithm(objective, mijt, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')

# %%
pop = [generation_xijt() for _ in range(n_pop)]
best, best_eval = 0, objective(decode(mijt, pop[0]))

for gen in range(n_iter):    
    decoded = [decode(mijt, p) for p in pop]
    # evaluate all candidates in the population
    scores = [objective(d) for d in decoded]

    # check for new best solution
    for i in range(n_pop):
        if scores[i] < best_eval:
            best, best_eval = pop[i], scores[i]

    # select parents
    selected = [selection(pop, scores) for _ in range(n_pop)]

    children = list()
    for i in range(0, n_pop, 2):
        # get selected parents in pairs
        p1, p2 = selected[i], selected[i+1]
        # crossover and mutation
        for c in crossover(p1, p2, r_cross):
            # mutation
            mutation(c, r_mut)
            # store for next generation
            children.append(bitstring2dict(c))
    pop = children
# %%
