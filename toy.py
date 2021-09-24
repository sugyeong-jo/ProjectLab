#%%
from mip import Model, xsum, minimize, BINARY, INTEGER

import pandas as pd

#%%
cost = pd.read_csv('cost.csv', index_col=0)
demand = pd.read_csv('demand.csv', index_col=0)
prepaid = pd.read_csv('prepaid.csv', index_col=0)
#%%
# set
T = list(range(len(cost.index)))
I = list(cost.columns)
J = ['A', 'B', 'C']
cit = {}
pit = {}
dit = {}
mijt = {}
#%%
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
#%%
pit
#%%
dit
#%%
mijt
#%%

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
        m += (uit[i, t] + xsum(xijt[i, j, t] for j in J) ==  dit[i, t])

# Constraint 2:-
for j in J:
    for t in T:
        m += (xsum(mijt[i, j, t]*xijt[i, j, t] for i in I) <=  1200)

# Constraint 3, 4:-
for i in I:
    for j in J:
        for t in T:
            m += (xijt[i, j, t] >= 0)
for i in I:
    for t in T:
        m += (uit[i, t] >= 0)

m.write('model.lp')
m.read('model.lp')
# %%
m.optimize()

# %%
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
# %%
m.objective_value
# %%
