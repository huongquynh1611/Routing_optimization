
import cplex
from cplex.exceptions import CplexError
import sys
import pandas as pd
import numpy as np
M = 40
xls = pd.ExcelFile('D:\\Quynh\\Routing_optimization\\modeling\\logistic.xlsx')
try:
    myProblem= cplex.Cplex()
    num_vehicle = 4
    num_location = 10 # tính cả depot
    my_names_ = ["x"+str(i)+str(j) + str(k) for k in range(1,num_vehicle + 1) for i in range(0,num_location) for j in range(0, num_location) ]  
    print(len(my_names_))
    # myProblem.variables.add( )    
except CplexError:
    print ("exc")

cost_matrix = pd.read_excel(xls, 'Cost Matrix')
my_obj = [float(cost_matrix.iat[i,j]) for i in range(0,num_location ) for j in range(1,num_location + 1 )]*num_vehicle


myProblem.variables.add(obj = my_obj,names= my_names_)
myProblem.objective.set_sense(myProblem.objective.sense.minimize)
num_var = len(my_names_)
# print(num_var)
for i in range(num_var):
    myProblem.variables.set_types(i,myProblem.variables.type.binary)

time = ["s"+str(i)+str(k) for k in range(1,num_vehicle+1) for i in range(0,num_location)]

myProblem.variables.add(names=time)
# set constraints:
constraints = list()  # list constraints includes var and coef
#   CONSTRAINTS (2) ONLY ONE VEHICLE SERVICE ONE CUSTOMER ( CUSTOMER 1,2,..,9)
names_1 = ["x"+str(i)+str(j) + str(k) for i in range(0, num_location) for k in range(1,num_vehicle+1)  for j in range(0, num_location)  ]
names_1 = np.array(names_1).reshape(num_location,(num_location*num_vehicle))  # (10,40)

# print(names_1[1:names_1.shape[0]])
for i in range(1,names_1.shape[0]):
    constraints.append([names_1[i],[1.0]*names_1.shape[1]])
# CONSTRAINTS (3) THE CAPICITY CONSTRAINTS FULFILLED
names_2 =  ["x"+str(i)+str(j) + str(k) for k in range(1,num_vehicle+1)    for j in range(0, num_location) for i in range(1, num_location)   ]
names_2 = np.array(names_2).reshape(num_vehicle,num_location*num_location - num_location)

demand = pd.read_excel(xls, 'Demand Matrix')['Value']
demand = [float(demand.iat[i]) for i in range(0,len(demand))]
# print(len(demand))
capity = pd.read_excel(xls, 'Capicity')['Value']

for i in range(0,names_2.shape[0]):
    constraints.append([names_2[i], demand*(int(names_2.shape[1] / len(demand)))])

# CONSTRAINTS (4) 
names_3 =np.array(names_1[[0]]).reshape(num_vehicle,num_location)
# print(names_3)
for i in range(0,names_3.shape[0]):
    constraints.append([names_3[i], [1]*names_3.shape[1]])

# CONSTRAINTS (5)
names_4 = ["x"+str(i)+str(j) + str(k)     for j in range(1, num_location) for k in range(1,num_vehicle+1) for i in range(0, num_location)   ]
# print(names_3)
names_5 = ["x"+str(i)+str(j) + str(k)  for i in range(1, num_location)    for k in range(1,num_vehicle+1)  for j in range(0, num_location)  ]
# print(names_4)
print(len(names_4))
names_4_5 = list()
for i in range(0,len(names_4)):
    if names_4[i] != names_5[i]:
        names_4_5.append(names_4[i] )
        names_4_5.append(names_5[i])


names_4_5 = np.array(names_4_5).reshape(int(len(names_4)/num_location),int(len(names_4_5)/(len(names_4)/num_location)))
# print(names_4_5)
for i in range(0,names_4_5.shape[0]):vê
    constraints.append([names_4_5[i], [1,-1]*(int(names_4_5.shape[1]/2))])
 
# CONSTRAINTS (6)
names_6 = ["x"+str(i)+'0' + str(k)   for k in range(1,num_vehicle+1) for i in range(1, num_location)   ]
names_6 = np.array(names_6).reshape(num_vehicle,num_location-1)
# print(names_6)
for i in range(0,names_6.shape[0]):
    constraints.append([names_6[i],[1]*names_6.shape[1]])
# CONSTRAINTS (10)    X_IJK = 0 IF I=J , FOR ALL I,J
names_10 = ["x" +str(i) + str(i) + str(k)  for i in range(0, num_location)    for k in range(1,num_vehicle+1)]
print(names_10)
print(len(names_10))
constraints.append([names_10,[1]*len(names_10)])
# CONSTRAINTS 11 X_IJK ! X_JIK
names_11



my_sense = ["E"]*(names_1.shape[0]-1) + ["L"]*(names_2.shape[0]) + ["E"] * (names_3.shape[0]) + ["E"] *names_4_5.shape[0] + ["E"]*names_6.shape[0] + ["E"]
    
my_rhs = [1]*(names_1.shape[0]-1) + [float(capity.iat[i]) for i in range(0,names_2.shape[0])] + [1]*names_3.shape[0] + [0] *names_4_5.shape[0] + [1]*names_6.shape[0] +[0]

my_rownames = ["c"+str(i) for i in range(1,len(my_rhs)+1)]  
myProblem.linear_constraints.add(lin_expr = constraints, senses = my_sense, rhs = my_rhs, names = my_rownames)
myProblem.solve()
try:
    a=myProblem.solution.get_values()
    b=myProblem.variables.get_names()
    for i in range(len(a)):
        if a[i] != 0:
            print((b[i],a[i]))
except cplex.exceptions.errors.CplexSolverError:
    print("No solution")



