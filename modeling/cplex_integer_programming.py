import cplex
from cplex.exceptions import CplexError
import sys
import pandas as pd
import numpy as np

'''
Maximize 52x1+30x2+20x3 object to the following constraints:
    2x1+4x2+5x3 <=100
    x1+x2+x3 <=30
    10x1+5x2+2x3 <=204
    x1,x2,x3 >=0 integer
'''
def ex1():
    try:
        prob= cplex.Cplex()
    except CplexError:
        print ("exc")
        return
    prob.objective.set_sense(prob.objective.sense.maximize)
    prob.variables.add(ub =[cplex.infinity,cplex.infinity,cplex.infinity] , obj= [52,30,20], names=['x' + str(i) for i in range(1,4)],types=[prob.variables.type.integer,prob.variables.type.integer,prob.variables.type.integer] )
    rows = [[["x1","x2","x3"],[2, 4,5]],[["x1","x2","x3"],[ 1,1,1]],[["x1","x2","x3"],[ 10,5,2]]]
        # print(rows)
    prob.linear_constraints.add(lin_expr = rows, senses = "LLL", rhs = [100,30,204], names =["c1","c2","c3"])
    # Solve the problem
    prob.solve()

    # And print the solutions
    print(prob.solution.get_values())
def ex2():
    xls = pd.ExcelFile('C:\\Users\\tlhqu\\Desktop\\Book1.xlsx')
    
    my_names_ = ["x_"+str(i)+str(j) for j in range(1, 9) for i in range(1, 5)]
    my_names_.sort()
    my_names = np.array(my_names_).reshape(4,8)

    print(my_names_)

    my_obj = np.array( pd.read_excel(xls, 'Cost')).flatten()
    print(my_obj)
  
    my_rhs_1 =  pd.read_excel(xls, 'Warehouse Supply')
    my_rhs_1 = my_rhs_1['Value'].to_numpy()
    print(my_rhs_1)
    my_sense_1 = "LLLL"
    my_ub = [cplex.infinity]*32
    my_rownames = ['c1','c2','c3','c4']
    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.minimize)
    
    prob.variables.add(obj = my_obj, ub = my_ub, names = my_names_,types=[prob.variables.type.integer]*32)

    rows = [[my_names[0],[1]*8],[my_names[1],[1]*8], [my_names[2],[1]*8],[my_names[3],[1]*8] ]
    prob.linear_constraints.add(lin_expr = rows, senses = my_sense_1, rhs = my_rhs_1, names = my_rownames)
    prob.solve()
def ex3():

    num_decision_var = 2
    num_constraints = 2

    A = [
        [1, 9/14],
        [-2, 1]
    ]
    b = [51/14, 1/3]
    c = [1, 1]

    constraint_type = ["L", "L"] # Less, Greater, Equal
    # ============================================================

    # Establish the Linear Programming Model
    myProblem = cplex.Cplex()

    # Add the decision variables and set their lower bound and upper bound (if necessary)
    myProblem.variables.add(names= ["x"+str(i) for i in range(num_decision_var)])
    for i in range(num_decision_var):
        myProblem.variables.set_lower_bounds(i, 0.0)
        a = myProblem.variables.get_lower_bounds()
        print(a)

    # Set the type of each variables
    myProblem.variables.set_types(0, myProblem.variables.type.integer)
    myProblem.variables.set_types(1, myProblem.variables.type.continuous)

    # Add constraints
    for i in range(num_constraints):
        myProblem.linear_constraints.add(
            lin_expr= [cplex.SparsePair(ind= [j for j in range(num_decision_var)], val= A[i])],
            rhs= [b[i]],
            names = ["c"+str(i)],
            senses = [constraint_type[i]]
        )

    # Add objective function and set its sense
    for i in range(num_decision_var):
        myProblem.objective.set_linear([(i, c[i])])
    myProblem.objective.set_sense(myProblem.objective.sense.maximize)

    # Solve the model and print the answer
    myProblem.solve()
    # print(myProblem.solution.get_values())
def ex4():
    xls = pd.ExcelFile('C:\\Users\\tlhqu\\Desktop\\Book1.xlsx')
    num_col = 4
    num_row = 2
    try:
        myProblem= cplex.Cplex()
        my_names_ = ["x_"+str(i)+str(j) for j in range(1, num_col + 1) for i in range(1, num_row + 1)]
        my_names_.sort()
        print(my_names_)
        myProblem.variables.add(names= my_names_ )    
    except CplexError:
        print ("exc")
        return
    # set sense for object
    myProblem.objective.set_sense(myProblem.objective.sense.minimize)
    # set avariable:
    num_var = len(my_names_)
    for i in range(num_var):
        myProblem.variables.set_upper_bounds(i, cplex.infinity)
    cost =  pd.read_excel(xls, 'Cost')
    
    my_obj = [float(cost.iat[i,j]) for i in range(0,num_row ) for j in range(0,num_col )]
    for i in my_obj:
        print(type(i))
    print(my_obj)
    my_ub = [cplex.infinity]*8
    print(my_ub)
    for i in range(num_var):
        myProblem.variables.set_types(i, myProblem.variables.type.integer)
    my_types = [myProblem.variables.type.integer]*8
    print(my_types)
    myProblem.variables.add(obj = my_obj)

    my_sense = "LLGGGG"
    supply = pd.read_excel(xls, 'Warehouse Supply')['Value']
    demand = pd.read_excel(xls, 'Customer Demand')['Value']
    my_rhs = [float(supply.iat[i]) for i in range(0,2)]+[float(demand.iat[i]) for i in range(0,4)]
    my_rownames = ["c"+str(i) for i in range(1,len(my_rhs)+1)]
    rows = [[my_names_[:4], [1,1,1,1]], [ my_names_[4:8],[1,1,1,1]] , [[my_names_[0],my_names_[4]],[1,1] ] , [[my_names_[1],my_names_[5]],[1,1] ] , [[my_names_[2],my_names_[6]],[1,1] ] , [[my_names_[3],my_names_[7]],[1,1] ] ]
    print(rows)
    myProblem.linear_constraints.add(lin_expr = rows, senses = my_sense, rhs = my_rhs, names = my_rownames)
    myProblem.solve()
    print(myProblem.solution.get_values())
def ex5():
    try:
        myProblem= cplex.Cplex()
        my_names_ = ["x_"+str(i) for i in range(1, 7)]
        print(my_names_)
        myProblem.variables.add(names= my_names_ )    
    except CplexError:
        print ("exc")
        return
    myProblem.objective.set_sense(myProblem.objective.sense.minimize)
    for i in range(6):
            myProblem.variables.set_lower_bounds(i, 0.0)
    for i in range(6):
        myProblem.variables.set_types(i, myProblem.variables.type.integer)
    myProblem.variables.set_upper_bounds(0, 1)
    myProblem.variables.set_upper_bounds(1, 1)
    myProblem.variables.set_upper_bounds(2, 1)
    myProblem.variables.set_upper_bounds(3, cplex.infinity)
    myProblem.variables.set_upper_bounds(4, cplex.infinity)
    myProblem.variables.set_upper_bounds(5, cplex.infinity)
    A = [[0,0,0,1,1,1],[-21000,0,0,1,0,0],[0,-20000,0,0,1,0],[0,0,-19000,0,0,1]]
    # A = np.array(A)
    rows = []
    for i in range(4):
        rows.append([my_names_,A[i]])
    print(rows)
    my_obj = [340000,270000,290000,32,33,30] 
    myProblem.variables.add( obj= my_obj)
    my_sense = ["G","L","L","L"]
    my_rhs = [38000,0,0,0]
    my_rownames = ["c"+str(i) for i in range(1,len(my_rhs)+1)]
    myProblem.linear_constraints.add(lin_expr = rows, senses = my_sense, rhs = my_rhs, names = my_rownames)
    myProblem.solve()
    print(myProblem.solution.get_values())
if __name__ == '__main__':
    ex5()