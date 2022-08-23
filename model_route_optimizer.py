from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
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
        self.model.TDistanceTime = namedtuple('TDistanceTime',["cus_from",'cus_to','distance','time'])
        self.model.TCostLateness   = namedtuple('TCostLateness',['customerid','late_delivery_cost'])
        self.model.TOrder = namedtuple('TOrder',['customerid','earliest_delivery','lastest_delivery','volume'])
        self.model.TDriver   = namedtuple('TDriver',['driverid','trucksize'])
        self.model.TTruckSize   = namedtuple('TTruckSize',['trucksize','capacity','quantity'])       
        self.model.TRoster = namedtuple('TRoster',['driverid','starttime','endtime'])
        # tuple for variables 
        
        return
    def read_input_data(self):
        file = open(self.json_input,'r')
        contents = file.read()
        file.close()
        self.input_json = json.loads(contents)
    def load_input_data(self):
        self.df_distancetime = json_normalize(self.input_json['Distance_time'])
        self.df_costlateness = json_normalize(self.input_json['UnitCostLateness'])
        self.df_wait_cost  = json_normalize(self.input_json['DriverWaitTimeCost'])
        self.df_order    = json_normalize(self.input_json['OrderDetails']) 
        self.df_driver    = json_normalize(self.input_json['Driver'])
        self.df_trucksize    = json_normalize(self.input_json['Truck size'])  
        self.df_roster   = json_normalize(self.input_json['Roster_Profile'])
      
        # convert to datetime format 
        self.anchor_date      = pd.to_datetime(self.df_roster['Start_time'].min() )
        self.df_order['EarliestDeliveryTime'] = pd.to_datetime(self.df_order['EarliestDeliveryTime'])
        self.df_order['LatestDeliveryTime'] = pd.to_datetime(self.df_order['LatestDeliveryTime'])
        self.df_roster['Start_time'] = pd.to_datetime(self.df_roster['Start_time'])
        self.df_roster['End_time'] = pd.to_datetime(self.df_roster['End_time'])

    def _convert_dt_to_num(self):
        def _convert_time_to_num(df: pd.DataFrame, col):
            "Applying date2num to an entire column"
            df.loc[:, col] = df[col].map(self.date_to_num)
            
        _convert_time_to_num(
            self.df_order, 'EarliestDeliveryTime')
        _convert_time_to_num(
            self.df_order, 'LatestDeliveryTime')
        _convert_time_to_num(
            self.df_roster, 'Start_time')
        _convert_time_to_num(
            self.df_roster, 'End_time')
    def load_model_object(self):
        self.vehicle_speed = 50 # km/h 
        self.unload_rate = 100 # units per hours
        self._convert_dt_to_num()
        
        self.wait_time_cost = self.df_wait_cost['DriverWaitTimeCost'][0]
       
        DISTIME       = [self.model.TDistanceTime(*row) for _,row in self.df_distancetime.iterrows()]
        COSTLATENESS  = [self.model.TCostLateness(*row) for _,row in self.df_costlateness.iterrows()]
        ORDER        = [self.model.TOrder(*row) for _,row in self.df_order.iterrows()]
        DRIVER        = [self.model.TDriver(*row) for _,row in self.df_driver.iterrows()]
        TRUCK         = [self.model.TTruckSize(*row) for _,row in self.df_trucksize.iterrows()]
        ROSTER        = [self.model.TRoster(*row) for _,row in self.df_roster.iterrows()] 
        
        
        
        self.disancetime = DISTIME 
        self.costlateness = COSTLATENESS 
        self.order = ORDER
        self.driver = DRIVER 
        self.truck = TRUCK 
        self.roster = ROSTER
    

        self.get_dist_time = {(t.cus_from,t.cus_to):t.time for t in self.disancetime}
        self.get_order_volume = {t.customerid:t.volume for t in self.order}
        

        # Object for driver-deliveries
        self.unique_cust_list = list(self.df_distancetime.cus_to.unique())
        self.unique_cust_list.remove('DEPOT01')
        self.driver_list = list(self.df_driver.driver_id.unique())
        
        #Object for truck-driver assignment
        self.driver_truck_assign = pd.merge(self.df_driver,self.df_trucksize,left_on='qualification',
                                                  right_on='Truck').drop('qualification',axis=1)
    
       
        self.truck_get = defaultdict(lambda:None,{t.trucksize:t for t in self.truck})
        
    


    def date_to_num(self, dt:datetime):
        "Convert date to numeric"
        return int((dt - self.anchor_date).total_seconds() / 3600)

    def num_to_datetime(self, hour):
        "Convert hour(int) to datetime based on anchor date"
        beginning_date = self.anchor_date
        i = math.floor(hour / 24)
        j = hour - 24 * i
        date = beginning_date + timedelta(days=i)
        date_time = date + timedelta(hours=j)
        return date_time
    
    
    def is_overlap(self, a, b, r=0):
        """
        a: tuple of range, ex (1,5)
        b: tuple of range, ex (4,10)
        r: overlap size, ex 1
        return True
        """
        return a[0] < b[1]+r and b[0] < a[1]+r
    def get_orders_by_range(self, start_time, end_time):
        a = (start_time, end_time)
        return [order 
                for order in self.order 
                if self.is_overlap(a, (order.earliest_delivery, order.lastest_delivery))]
    
    
        
class SetupData(ModelObjects): 
    def __init__(self,data,model: Model):
        self.json_input = data
        self.model = model 
    def create_model_objects(self):
        self.read_input_data()
        self.load_input_data()
        self.create_tuples()
        self.load_model_object()
        self.create_var_route()
    
        
 
    def create_var_route(self):
        route_var = []
        for roster in self.roster:
            
            order_avai = self.get_orders_by_range(roster.starttime, roster.endtime) 
            for order_from in order_avai: 
                for order_to in order_avai: 
                    if order_from==order_to:
                        continue
                    var = self.model.binary_var()
                    waiting_time = self.model.continuous_var() 
                    arrival_time = self.model.continuous_var() 
                    loading_vehicle = self.model.continuous_var() 
                    volume = roster.volume
                    customerid = roster.customerid
                    earliest_time = roster.earliest_delivery 
                    latest_time = roster.lastest_delivery
                    
        

        return
        
    

    
class SetupConstraints(ModelObjects):
    def __init__(self,data,model:Model):
        self.input_json = data 
        self.model = model
    

    def add_constraints(self):
        
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






























