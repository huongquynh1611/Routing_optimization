
import cplex
from cplex.exceptions import CplexError
import sys
import pandas as pd
import numpy as np
M = 40
xls = pd.ExcelFile('D:\\Quynh\\Routing_optimization\\modeling\\logistic.xlsx')
try:
    myProblem= cplex.Cplex()
    num_vehicle = 3
    num_location = 10# tính cả depot
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
for i in range(0,names_4_5.shape[0]):
    constraints.append([names_4_5[i], [1,-1]*(int(names_4_5.shape[1]/2))])
 
# CONSTRAINTS (6)
names_6 = ["x"+str(i)+'0' + str(k)   for k in range(1,num_vehicle+1) for i in range(1, num_location)   ]
names_6 = np.array(names_6).reshape(num_vehicle,num_location-1)
# print(names_6)
for i in range(0,names_6.shape[0]):
    constraints.append([names_6[i],[1]*names_6.shape[1]])



# CONSTRAINTS (10)    X_IJK = 0 IF I=J , FOR ALL I,J
names_10 = ["x" +str(i) + str(i) + str(k)  for i in range(0, num_location)    for k in range(1,num_vehicle+1)]
# print(names_10)
# print(len(names_10))
constraints.append([names_10,[1]*len(names_10)])
# CONSTRAINTS 11 X_IJK + X_JIK <=1
names_11 = ["x" + str(i) + str(j) + str(k) for i in range(0,num_location) for j in range(0,num_location) for k in range(1,num_vehicle+1) if i!=j]
names_12 = ["x" + str(j) + str(i) + str(k) for i in range(0,num_location) for j in range(0,num_location) for k in range(1,num_vehicle+1) if i!=j]
# print(names_11)
# print(names_12)
names_13 = list()
for i in range(len(names_11)):
    names_13.append(names_11[i])
    names_13.append(names_12[i])
names_13=  np.array(names_13).reshape(int(len(names_13)/2),2)
for i in range(0,names_13.shape[0]):
    constraints.append([names_13[i],[1,1]])

print(names_13)
# CONSTRAINTS (7) MAKE SURE VEHICLE SHoULD NOT ARRIVE AT CUS J BEFORE TIME ...

names_8 = ["s" + str(i ) + str(k)for k in range(1,num_vehicle+1) for i in range(0,num_location) ]
names_8 = [i for i in names_8 for j in range(num_location)] 
# print((names_8))
names_9 = ["s" + str(i ) + str(k)for k in range(1,num_vehicle+1) for i in range(0,num_location) ]
names_9 = [names_9[i:i+num_location] for i in range(0,len(names_9),num_location) for j in range(num_location)]
names_9 =  [i for j in names_9 for i in j]
# print(names_8)
# print(my_names_)
# print((names_9))
names_7 = list()
for i in range(0,len(names_8)):
    names_7.append(names_8[i])
    names_7.append(my_names_[i])
    names_7.append(names_9[i])
names_7 = np.array(names_7).reshape(int(len(names_7)/3),3)

import collections

new_names_7 = list()
for i in names_7:
    new = [item for item,count in collections.Counter(i).items() if count > 1]
#     print(new)
    if len(new) == 0:
        new_names_7.append(list(i))
new_names_7 = np.array(new_names_7)   

for i in range(len(new_names_7)):
    constraints.append([  new_names_7[i],[1,M,-1]*len(new_names_7)])
time_travel = pd.read_excel(xls, 'Time Matrix')
time_travel = [-float(time_travel.iat[i,j])  for i in range(0,num_location ) for j in range(1,num_location+1 )]
for i in time_travel:
    if (i == float(0)):
        time_travel.remove(i)
time_travel = time_travel*num_vehicle
# CONSTRAINTS (8)
for i in range(0,len(time)):
    constraints.append([[time[i]],[1]])     
for i in range(0,len(time)):
    constraints.append([[time[i]],[1]])
start_time = pd.read_excel(xls, 'Time Window')['Time Start']
   
end_time = pd.read_excel(xls, 'Time Window')['Time End']
print(len(time))


# SOLVING
my_sense = ["E"]*(names_1.shape[0]-1) + ["L"]*(names_2.shape[0]) + ["E"] * (names_3.shape[0]) + ["E"] *names_4_5.shape[0] + ["E"]*names_6.shape[0] + ["E"] + ["L"]*names_13.shape[0] +["L"]*len(new_names_7) + ["L"]*len(time)+ ["G"]*len(time)
    
my_rhs = [1]*(names_1.shape[0]-1) + [float(capity.iat[i]) for i in range(0,names_2.shape[0])] + [1]*names_3.shape[0] + [0] *names_4_5.shape[0] + [1]*names_6.shape[0] +[0] + [1]*names_13.shape[0] + time_travel + [float(end_time.iat[i]) for i in range(0,num_location)]*num_vehicle + [float(start_time.iat[i]) for i in range(0,num_location)]*num_vehicle

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



