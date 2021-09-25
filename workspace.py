from mip import Model, xsum, minimize, BINARY, INTEGER

import pandas as pd

# variables infomation
T = list(set(df_sample.T))
I = list(set(df_sample.item))
J = list(set(df_sample.J))

# Parameters
for index, info in cost1.itterows():
    cit[info['item'], info['T']] = info['C']
for index, info in certi1.iterrows():
    pit[info['item'], info['T']] = info['P']
for index, info in demand1.iterrows():
    dit[info['item'], info['T']] = info['D']
for index, info in cycle1.iterrows():
    mijt[info['item'], info['J'], info['T']] = info['M']

#variables
m = Model("Unist_optimization")
xijt = {
    (i, j, t): m.add_var(var_type=INTEGER, name='x_%s,%s,%s' % (i, j, t))
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
