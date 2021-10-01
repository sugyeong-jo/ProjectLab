#%%
from mip import Model, xsum, minimize, BINARY, INTEGER

import pandas as pd
pd.set_option('display.max_row', 500)

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
    machine_bound[item, machine] = info['AVG_CT']
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

#%%
machine_bound
#%%

# %%


def generation_xijt():
    xijt = {}
    for i in I:
        for j in J:
            for t in T:
                if mijt[i, j, t] == 0:
                    xijt[i, j, t] = 0
                else:
                    xijt[i, j, t] = randint(0, 300)
    return xijt

def generation_xijt():
    xijt = {}
    for i in I:
        for j in J:
            for t in T:
                # if mijt[i, j, t] == 0:
                #     xijt[i, j, t] = 0
                # else:
                xijt[i, j, t] = randint(0, 300)
    return xijt

def decode(mijt, xijt):
    mijt = dict2bitstring(mijt,)
    xijt = dict2bitstring(xijt)
    for i, value in enumerate(xijt):
        ub = mijt[i]
        if value > ub:
            if ub == 0:
                xijt[i] = 0
            else:
                xijt[i] = randint(0,300 ) # mijt[i]
    xijt = bitstring2dict(xijt, 'xijt')
    return xijt

def decode(mijt, xijt):
    #mijt = dict2bitstring(mijt)
    #xijt = dict2bitstring(xijt)
    check_list = []
    for j in J:
        for t in T:
            check = sum(mijt[i, j, t]*xijt[i, j, t] for i in I) <=  600
            if check == False:
                for i in I:
                    if mijt[i, j, t] == 0:
                        xijt[i, j, t] = 0
                    else:
                        xijt[i, j, t] = randint(0, 300)
                        #print(math.ceil(600/mijt[i, j, t]))
    for i in I:
        for j in J:
            for t in T:
                if xijt[i, j, t] < 0:
                    xijt[i, j, t] = 0
    return xijt

def dict2bitstring(xijt):
    return list(xijt.values())


def constraint_check(xijt):
    #  Constraint 2
    check_list = []
    for j in J:
        for t in T:
            check = sum(mijt[i, j, t]*xijt[i, j, t] for i in I) <=  600
            check_list.append(check)
    const_2 = all(i==True for i in check_list)
    return const_2 == True


def bitstring2dict(bitstring, type='xijt'):
    if type == 'xijt':
        _keys = xijt_keys
    elif type == 'mijt':
        _keys = mijt_keys

    for idx, value in enumerate(bitstring):
        xijt[_keys[idx]] = value
    return xijt


def objective(xijt):
    uit = {}
    if constraint_check(xijt) == False:
        return 100000000
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
def selection(pop, scores, k=5):
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


def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    pop = [generation_xijt() for _ in range(n_pop)]
    best, best_eval = decode(mijt, pop[0]), objective(decode(mijt, pop[0]))
    print(best_eval)

    for gen in range(n_iter):
        decoded = [decode(mijt, p) for p in pop]
        # evaluate all candidates in the population
        scores = [objective(d) for d in decoded]

        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(f'>{gen}, {scores[i]}')

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
check_list = decode(mijt, best)
for i in I:
    for j in J:
        for t in T:
            value = check_list[i, j, t] - xijt_oracle[i, j, t]
            if value > 0:
                print(i,j, t, int(value), int(check_list[i, j, t]) , int(xijt_oracle[i, j, t]))
#%%
check_list = decode(mijt, xijt)
for i in I:
    for j in J:
        for t in T:
            if check_list[i, j, t] < 0:
                print(i,j, t, value,  check_list[i, j, t])
#%%
i = 9
dict2bitstring(xijt_oracle)[i]
#%%
list(xijt_oracle)[i]
#%%
xijt_oracle[list(xijt_oracle)[i]]
#%%
mijt[list(xijt_oracle)[i]]
#%%
list(xijt)[i]
#%%
objective(decode(mijt, xijt_oracle))
#%%
pop = [generation_xijt() for _ in range(n_pop)]
objective(decode(mijt, pop[0]))
pop[0]
decode(mijt, pop[0])
#%%
mijt['K04046', 424, 1]

#%%
decode(mijt, xijt)['K04046', 424, 1]

#%%
xijt_oracle['K04046', 424, 1]

#%%
n_iter = 50
# bits per variable
n_bits = 16
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / (float(n_bits) * len(xijt))

xijt = generation_xijt()
xijt_keys = list(xijt.keys())
mijt_keys = list(mijt.keys())

best, score = genetic_algorithm(objective, mijt, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
#%%
best
# %%


n_iter = 500
# bits per variable
n_bits = 16
# define the population size
n_pop = 500
# crossover rate
r_cross = 0.3
# mutation rate
r_mut = 1.0 / (float(n_bits) * len(xijt))

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
            print(f'>{gen}, {scores[i]}')

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
best
# %%
solution
# %%
xijt_oracle = {}

for index, info in solution[solution['variable'].str.contains('x_')].iterrows():
    index_info = info['variable'][2:].split(',')
    xijt_oracle[index_info[0], int(index_info[1]), int(index_info[2])]=info['solution']
# %%
objective(xijt_oracle)

# %%
m.objective_value

# %%
uit_check = {}

for i in I:
    for t in T:
        u = dit[i, t] - sum(xijt[i, j, t] for j in J)
        if u >= 0:
            uit_check[i, t] = u
        else:
            uit_check[i, t] = 0

# %%

len(uit_check)

uit_check
dit['K04046', 2]
sum(xijt_oracle['K04046', j, 2] for j in J)


# %%
solution[solution['variable'].str.contains('u_K04046')]

#%%
solution[solution['variable'].str.contains('x_K04046')]

# %%
uit_check['K04031', 1]

# %%
dit['K04031', 1]
# %%
sum(xijt_oracle['K04031', j, t] for j in J for t in T)

#%%
xijt_oracle['K04031', 1, 5] 
# %%
for t in T:
    for j in J:
        print(xijt_oracle['K04101', j, t])
# %%
sum(solution[solution['variable'].str.contains('x_K04101')]['solution'])
# %%
J
# %%
solution[solution['variable'].str.contains('x_K04101')]['solution']
# %%

dit
# %%
