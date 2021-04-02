
import cplex
from cplex.exceptions import CplexError
import sys
import pandas as pd
import numpy as np
def ex1():
    xls = pd.ExcelFile('C:\\Users\\tlhqu\\Desktop\\logistic.xlsx')
    num_col=3
    num_row = 7
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
    
    my_obj = [float(cost.iat[i,j]) for i in range(0,num_row ) for j in range(1,num_col + 1 )]
    print(my_obj)
    for i in range(num_var):
        myProblem.variables.set_types(i, myProblem.variables.type.integer)
    myProblem.variables.add(obj = my_obj)
    
    # set constraints
    my_sense = "GGGGGGGLLL"
    demand = pd.read_excel(xls, 'Demand')['Value']
    storage = pd.read_excel(xls, 'Storage')['Value']
    my_rhs = [float(demand.iat[i]) for i in range(0,7)] + [float(storage.iat[i]) for i in range(0,3)]
    print(my_rhs)
  
    my_rownames = ["c"+str(i) for i in range(1,len(my_rhs)+1)]
    rows = [[my_names_[:3], [1,1,1]], [ my_names_[3:6],[1,1,1]] , [my_names_[6:9],[1,1,1] ] , [my_names_[9:12],[1,1,1]] , [my_names_[12:15],[1,1,1]] ,[my_names_[15:18],[1,1,1] ],[my_names_[18:21],[1,1,1] ], [[my_names_[0],my_names_[3],my_names_[6],my_names_[9],my_names_[12],my_names_[15],my_names_[18]],[1,1,1,1,1,1,1]] , [ [my_names_[1],my_names_[4],my_names_[7],my_names_[10],my_names_[13],my_names_[16],my_names_[19]],[1,1,1,1,1,1,1]] ,[[my_names_[2],my_names_[5],my_names_[8],my_names_[11],my_names_[14],my_names_[17],my_names_[20] ],[1,1,1,1,1,1,1]] ]
    print(rows)
    myProblem.linear_constraints.add(lin_expr = rows, senses = my_sense, rhs = my_rhs, names = my_rownames)
    myProblem.solve()
    print(myProblem.solution.get_values())

def ex2():
    M = 40
    xls = pd.ExcelFile('C:\\Users\\tlhqu\\Desktop\\logistic.xlsx')
    try:
        myProblem= cplex.Cplex()
        num_vehicle = 4
        num_location = 10 # tính cả depot
        my_names_ = ["x"+str(i)+str(j) + str(k) for k in range(1,num_vehicle + 1) for i in range(0,num_location) for j in range(0, num_location) ]  
        print(len(my_names_))
        # myProblem.variables.add( )    
    except CplexError:
        print ("exc")
        return
    cost_matrix = pd.read_excel(xls, 'Cost Matrix')
    my_obj = [float(cost_matrix.iat[i,j]) for i in range(0,10 ) for j in range(1,11 )]*4
    
    print(len(my_obj))
    myProblem.variables.add(obj = my_obj,names= my_names_)
    myProblem.objective.set_sense(myProblem.objective.sense.minimize)
    num_var = len(my_names_)
    # print(num_var)
    for i in range(num_var):
        myProblem.variables.set_types(i,myProblem.variables.type.binary)
    
    time = ["s"+str(i)+str(k) for k in range(1,5) for i in range(0,10)]
    myProblem.variables.add(names=time)

     # set constraints
    
    names_1 = ["x"+str(i)+str(j) + str(k) for i in range(0, 10) for k in range(1,5)  for j in range(0, 10)  ]
    names_1 = np.array(names_1).reshape(10,40)

    # print(names[0].reshape(4,10)[0])
    constraints = list()
    for i in range(1,10):
        constraints.append([names_1[i],[1.0]*40])
    # print(constraints)
    for i in range(0,4):
        constraints.append([names_1[0].reshape(4,10)[i],[1.0]*10])
    names_2 =  ["x"+str(i)+str(j) + str(k) for k in range(1,5)    for j in range(0, 10) for i in range(1, 10)   ]
    names_2 = np.array(names_2).reshape(4,90)
    # print(names_3)
    demand = pd.read_excel(xls, 'Demand Matrix')['Value']
    demand = [float(demand.iat[i]) for i in range(0,8)]
    # print(demand)
    capity = pd.read_excel(xls, 'Capity')['Value']
    for i in range(0,4):
        constraints.append([names_2[i], demand*90])
    names_3 = ["x"+str(i)+str(j) + str(k)     for j in range(1, 10) for k in range(1,5) for i in range(0, 10)   ]

    names_4 = ["x"+str(i)+str(j) + str(k)  for i in range(1, 10)    for k in range(1,5)  for j in range(0, 10)  ]
    names_5 = list()
    for i in range(0,360):
        if names_3[i] != names_4[i]:
            names_5.append(names_3[i] )
            names_5.append(names_4[i])
    # print(names_5)
    names_5 = np.array(names_5).reshape(36,18)
    for i in range(0,36):
        constraints.append([names_5[i],[1.0,-1.0]*9 ])
    names_6 = ["x"+str(i)+'0' + str(k)     for k in range(1,5) for i in range(1, 10)   ]
    names_6 = np.array(names_6).reshape(4,9)
    # print(names_6)
    for i in range(0,4):
        constraints.append([names_6[i],[1]*9])
    for i in range(0,40):
        constraints.append([[time[i]],[1]])     
    for i in range(0,40):
        constraints.append([[time[i]],[1]])
    names_8 = ["s" + str(i ) + str(k)for k in range(1,5) for i in range(0,10) ]
    names_8 = [i for i in names_8 for j in range(10)] 
    # print((names_8))
    names_9 = ["s" + str(i ) + str(k)for k in range(1,5) for i in range(0,10) ]
    names_9 = [names_9[i:i+10] for i in range(0,len(names_9),10) for j in range(10)]
    names_9 =  [i for j in names_9 for i in j]
    # print(names_8)
    # print(my_names_)
    # print((names_9))
    names_7 = list()
    for i in range(0,400):
        names_7.append(names_8[i])
        names_7.append(my_names_[i])
        names_7.append(names_9[i])
    names_7 = np.array(names_7).reshape(400,3)
    print(names_7)

    import collections

    new_names_7 = list()
    for i in names_7:
        new = [item for item,count in collections.Counter(i).items() if count > 1]
    #     print(new)
        if len(new) == 0:
            new_names_7.append(list(i))
    new_names_7 = np.array(new_names_7)
    print(len(new_names_7))

    # names_10 = ['x' + str(i) + str(i) + str(k) for k in range(1,5) for i in range(0,10) ]
    # names_10 = np.array(names_10).reshape(40,1)
    for i in range(360):
        constraints.append([  new_names_7[i],[1,M,-1]*360])
    # for i in range(40):
    #     constraints.append([names_10[i],M])
    
    # print(names_10)
    # print(names_7)
    # print(len(names_7))
    
    time_travel = pd.read_excel(xls, 'Time Matrix')
    time_travel = [-float(time_travel.iat[i,j])  for i in range(0,10 ) for j in range(1,11 )]
    for i in time_travel:
        if (i == float(0)):
            time_travel.remove(i)
    time_travel = time_travel*4
    
    # print(time_travel)
    print(len(time_travel))
    start_time = pd.read_excel(xls, 'Time Window')['Time Start']
   
    end_time = pd.read_excel(xls, 'Time Window')['Time End']
    my_sense = ["E"]*13 + ["L"]*4 + ["E"] * 36 + ["E"]*4 + ["L"] *40 + ["G"] *40  +["L"]*360
    # print(len(my_sense))
    my_rhs = [1]*13 + [float(capity.iat[i]) for i in range(0,4)] + [0]*36 + [1]*4 + [float(end_time.iat[i]) for i in range(0,10)]*4 + [float(start_time.iat[i]) for i in range(0,10)]*4 + time_travel
    
    my_rownames = ["c"+str(i) for i in range(1,len(my_rhs)+1)]  
    myProblem.linear_constraints.add(lin_expr = constraints, senses = my_sense, rhs = my_rhs, names = my_rownames)
    myProblem.solve()
    # print(myProblem.solution.get_values())


if __name__ == '__main__':
    ex2()