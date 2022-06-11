import pandas as pd 
import math
import numpy as np
from datetime import timedelta
from collections import namedtuple
from collections import defaultdict
from docplex.mp.model import Model
from tqdm import tqdm
from datetime import datetime
import cplex
from docplex.mp.solution import SolveSolution
import getopt
import requests
import json 
import ast
from pandas import json_normalize
import pandas as pd
import sys
from docplex.mp.model_reader import ModelReader
import os
from os.path import dirname, abspath
import time


class ModelObjects:
    def __init__(self):
        self.VAR = []
    def create_tuples(self):
        self.model.TCapacity = namedtuple('TCapacity',["TruckID",'Capacity','Units','Depot_ID'])
        self.model.TDepot    = namedtuple('TDepot',['ID','Longitude','Latitude','TimeService'])
        self.model.TEmployee = namedtuple('TEmployee',['EmployeeID','Type','StartTime','EndTime','Depot_ID'])
        self.model.TOrder   = namedtuple('TOrder',['OrderID','StoreID','Case','Receipt_Time_Start','Receipt_Time_End'])
        self.model.TStore   = namedtuple('TStore',['ID','Longitude','Latitude','Time_Service'])       
        self.model.TConvert = namedtuple('TConvert',['From','To','Rate'])
        # tuple for variables 
        self.model.TROUTE   = namedtuple('TROUTE',['from_store','to_store','distance','vehicle_id','depot_id'])
        self.model.TRoute   = namedtuple('TRoute',['var','name','from_store','to_store','distance','day','vehicle_id','depot_id'])
        self.model.TTime    = namedtuple('TTime',['var','name','store','order_id','start_time','end_time','employee_id','type','case','depot_id']) 
        self.model.TVar     = namedtuple('TVar',['order_id','store_id',"receipt_start",'receipt_end','route','time','case'])
        return
    def read_input_data(self):
        file = open(self.json_input,'r')
        contents = file.read()
        file.close()
        self.input_json = json.loads(contents)
    def load_input_data(self):
        self.model.df_capacity = json_normalize(self.input_json['Capacity'])
        self.model.df_employee = json_normalize(self.input_json['Employee'])
        self.model.df_convert  = json_normalize(self.input_json['Convert'])
        self.model.df_order    = json_normalize(self.input_json['Order'])
        self.model.df_store    = json_normalize(self.input_json['Store'])
        self.model.df_depot    = json_normalize(self.input_json['Depot'])  
        self.model.df_store    = pd.concat([self.model.df_store,self.model.df_depot])
      
        # convert to datetime
        self.model.df_order['Receipt_Time_Start'] = pd.to_datetime(self.model.df_order['Receipt_Time_Start'],format='%Y-%d-%m %H:%M:%S')
        self.model.df_order['Receipt_Time_End'] = pd.to_datetime(self.model.df_order['Receipt_Time_End'],format='%Y-%d-%m %H:%M:%S')
        self.model.day_list = list(set(self.model.df_order.Receipt_Time_Start.dt.strftime('%Y-%m-%d')))
        
        self.model.df_employee['StartTime'] = pd.to_datetime(self.model.df_employee['StartTime'],format='%Y-%d-%m %H:%M:%S')
        self.model.df_employee['EndTime'] = pd.to_datetime(self.model.df_employee['EndTime'],format='%Y-%d-%m %H:%M:%S')
        self.model.df_order = self.model.df_order.sort_values(['Receipt_Time_Start','Receipt_Time_End'])
        self.model.anchor_date      = pd.to_datetime(self.model.df_employee['StartTime']).min()
        # convert to num
        self.list_day = list(set(self.model.df_order.Receipt_Time_Start.dt.strftime('%Y-%m-%d')))
        self.convert_time_to_num(self.model.df_order, 'Receipt_Time_Start')
        self.convert_time_to_num(self.model.df_order, 'Receipt_Time_End')
        self.convert_time_to_num(self.model.df_employee, 'StartTime')
        self.convert_time_to_num(self.model.df_employee, 'EndTime')
        
    def load_model_object(self):
        CAPACITY = [self.model.TCapacity(*row) for _,row in self.model.df_capacity.iterrows()]
        EMPLOYEE = [self.model.TEmployee(*row) for _,row in self.model.df_employee.iterrows()]
        CONVERT  = [self.model.TConvert(*row) for _,row in self.model.df_convert.iterrows()]
        ORDER    = [self.model.TOrder(*row) for _,row in self.model.df_order.iterrows()]
        STORE    = [self.model.TStore(*row) for _,row in self.model.df_store.iterrows()]
        DEPOT    = [self.model.TDepot(*row) for _,row in self.model.df_depot.iterrows()]
        
        self.model.capacity = CAPACITY 
        self.model.employee = EMPLOYEE 
        self.model.convert = CONVERT
        self.model.order = ORDER 
        self.model.store = STORE 
        self.model.depot = DEPOT
    def date2num(self, dt):
        return  int((dt - self.model.anchor_date).total_seconds() / 3600)
     
    def convert_time_to_num(self, df, col):
        df.loc[:, col] = list(map(self.date2num, df[col]))

    def set_depot(self):
        depot_id = self.model.df_depot.ID.values
        self.depot = dict( ( depot, (float(self.model.df_depot.loc[self.model.df_depot.ID == depot, 'Longitude'].values[0]), float(self.model.df_depot.loc[self.model.df_depot.ID == depot, 'Latitude'].values[0])) ) for depot in depot_id)

    def set_time(self):
        self.model.BEGIN_DATETIME = pd.to_datetime(self.model.df_order.Receipt_Time_Start).min()
    def convert_to_datetime(self,hour):
        beginning_date = self.model.anchor_date
        i = math.floor(hour / 24)
        j = int(hour - 24 * i)
        date = beginning_date + timedelta(days=i)
        date_time = date + timedelta(hours=j)
        return date_time
    def set_conversation_rate(self):
        self.RATE_conversation = self.model.df_convert.loc[(self.model.df_convert.From =='Pallets')]['Rate'].values[0]
    def get_distance(self,lat1,long1,lat2,long2):
        
        return (abs(math.sqrt((pow(lat1 - lat2,2))+ (pow(long1-long2,2)))))
        
    def get_position(self):
        self.positions = dict( ( store, (float(self.model.df_store.loc[self.model.df_store.ID == store, 'Longitude'].values[0]), float(self.model.df_store.loc[self.model.df_store.ID == store, 'Latitude'].values[0])) ) for store in self.model.df_store.ID)
        
    def get_all_route(self):
        all_route = []
        store_list = list(self.model.df_store.ID)
        vehicle_list = list(self.model.df_capacity.Truck_ID)
        for i in store_list:
            for j in store_list:
                for k in vehicle_list:
                    if i != j:
                        position_dict = self.positions 
                        long1 = position_dict[i][0]
                        lat1  = position_dict[i][1]
                        long2 = position_dict[j][0]
                        lat2 = position_dict[j][1]
                        distance = self.get_distance(lat1,long1,lat2,long2)
                        depot_id = list(self.model.df_capacity[self.model.df_capacity.Truck_ID == k]['Depot_ID'])[0]
                        if i not in ['Dep_01','Dep_02'] or ( i == depot_id):
                            all_route.append((i,j,distance,k,depot_id))
        self.model.df_all_route = pd.DataFrame(all_route,columns=['FromStore','ToStore','Distance','VehicleID','Depot_ID'])
        self.model.all_route = [self.model.TROUTE(*row) for _,row in self.model.df_all_route.iterrows()]
      
    def convert_to_pallet(self):
        return
        
class SetupData(ModelObjects):
    def __init__(self,data,model: Model):
        self.json_input = data
        self.model = model 
        self.model.route_obj = []
        self.model.time_obj = []
    def create_model_objects(self):
        self.read_input_data()
        self.load_input_data()
        self.create_tuples()
        self.load_model_object()
        self.set_time()
        self.set_conversation_rate()
        self.get_position()
        self.set_depot()  
        self.get_all_route() 
        self.create_route_var()   
        self.create_time_var()
   
   
        
    
    def create_route_var(self):
        
        self.ROUTE = []
        for day in self.model.day_list:
            for route in self.model.all_route:
                
                from_store = route.from_store
                to_store   = route.to_store
                vehicle    = route.vehicle_id
                distance   = route.distance 
                depot_id   = route.depot_id
                
                name = from_store + '_' + to_store + '_' + vehicle +'_route'
                var = self.model.binary_var()
                self.ROUTE.append((var,name,from_store,to_store,distance,day,vehicle,depot_id))
        self.model.route_obj = [self.model.TRoute(*row) for row in self.ROUTE]
        # C1: each vehicle drive from only 1 store 1 time a day 
        list_from_store_each_day = {} 
        for i in self.model.route_obj:
            key = i.from_store + i.day
            if not key in list_from_store_each_day: 
                list_from_store_each_day[key] = []
            list_from_store_each_day[key].append(i)
        self.model.list_from_store_each_day = list_from_store_each_day
        # C2: each vehicle drive to only 1 store 1 time a day 
        list_to_store_each_day = {} 
        for i in self.model.route_obj:
            key = i.to_store + i.day
            if not key in list_to_store_each_day: 
                list_to_store_each_day[key] = []
            list_to_store_each_day[key].append(i)
        self.model.list_to_store_each_day = list_to_store_each_day
        # C3: the number of order per store per day 
        list_number_order = {}
        for order in self.model.order: 
            key = str(self.convert_to_datetime( order.Receipt_Time_Start).date()) + '_' + order.StoreID
            if not key in list_number_order :
                list_number_order[key] = []
            list_number_order[key].append(order)
        for i in list_number_order:
            list_number_order[i] = len(list_number_order[i])
        for i in list_number_order: 
            print('------>',i,list_number_order[i])
        

        
         
    def create_time_var(self):
        return
        
    

    
class SetupConstraints(ModelObjects):
    def __init__(self,data,model:Model):
        self.input_json = data 
        self.model = model
    def route_constraint(self):
        # C1: each vehicle drive from only 1 store 1 time a day 
        list_from_store_each_day = self.model.list_from_store_each_day
        for i in list_from_store_each_day:
            self.model.add_constraint(self.model.sum(v.var for v in list_from_store_each_day[i] )   == 1)
        # C2: each vehicle drive to only 1 store 1 time a day 
        list_to_store_each_day = self.model.list_to_store_each_day 
        for i in list_to_store_each_day:
            self.model.add_constraint(self.model.sum(v.var for v in list_to_store_each_day[i]) == 1)
        
        
        
        return
            


    def add_constraints(self):
        self.route_constraint()
        print('Finished constraints')
class SetupObjectives(ModelObjects):
    def __init__(self,model: Model):
        self.model = model
class ModelSolver(ModelObjects):
    def __init__(self,model):
        self.model = model 
    def solve_model(self):
        solve=self.model.solve(log_output=True)
        print("Model solve complete")
        return solve
class CreateResults(ModelObjects):
    def __init__(self,data,model,solution): 
        self.input_json = data
        self.model = model
        self.solution = solution
class ModelBuild:
    def __init__(self,data):
        self.model = Model()
        self.model.start_time = time.time()
        self.data = data
    def create_model_run(self):
        self.SetupData = SetupData(self.data,self.model)
        self.SetupData.create_model_objects()
        self.Setup_constraints = SetupConstraints(self.data,self.model)
        self.Setup_constraints.add_constraints()
        self.Setup_objectives = SetupObjectives(self.model)
        self.ModelSolve = ModelSolver(self.model)
        self.solutoin = self.ModelSolve.solve_model()
        self.CreateResults = CreateResults(self.data,self.model,self.solutoin)
if __name__ == '__main__':
    start_time = time.time()
    mb=ModelBuild('data.json')
    mb.create_model_run()
    print("--- %s seconds ---" % (time.time() - start_time))






























