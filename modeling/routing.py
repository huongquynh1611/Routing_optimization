
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
my_obj = [float(cost_matrix.iat[i,j]) for i in range(0,num_location ) for j in range(1,num_location + 1 )]*4

print(len(my_obj))
myProblem.variables.add(obj = my_obj,names= my_names_)
myProblem.objective.set_sense(myProblem.objective.sense.minimize)
num_var = len(my_names_)
# print(num_var)
for i in range(num_var):
    myProblem.variables.set_types(i,myProblem.variables.type.binary)

time = ["s"+str(i)+str(k) for k in range(1,num_vehicle+1) for i in range(0,num_location)]

myProblem.variables.add(names=time)
print(myProblem.variables.get_names())