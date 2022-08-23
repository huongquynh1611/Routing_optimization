# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 09:21:11 2022

@author: Bronwyn
"""

from collections import namedtuple
import datetime
from sqlite3 import Timestamp
from tqdm import tqdm
import pandas as pd
from docplex.mp.model import Model
from docplex.util.environment import get_environment
from functools import reduce
import numpy as np
import uuid
import math
from datetime import timedelta
import sys
import time
import json
from pandas import json_normalize
import requests
import os
from collections import defaultdict
import ast

class ModelObjects:
    def __init__(self, data, model: Model):
        self.data = data
        self.model = model
        
    def read_input_data_excel(self):
        self.model.xlsx = pd.ExcelFile('Routing_input_new.xlsx')
        
    def load_input_data_excel(self):
        self.model.df_wait_cost = pd.read_excel(self.model.xlsx,'DriverWaitTimeCost')
        self.model.df_dist_time = pd.read_excel(self.model.xlsx,'Distance_time')
        self.model.df_lateness_cost = pd.read_excel(self.model.xlsx,'UnitCostLateness')
        self.model.df_order = pd.read_excel(self.model.xlsx,'OrderDetails')
        self.model.df_driver = pd.read_excel(self.model.xlsx,'Driver')
        self.model.df_truck_size = pd.read_excel(self.model.xlsx,'Truck size')
        self.model.df_roster = pd.read_excel(self.model.xlsx,'Roster_Profile')
        self.model.unload_rate = 100
        

    def create_namedtuples(self):
        # self.TWaitCost = namedtuple("TWaitCost", ['wait_cost'])
        self.TDistTime = namedtuple("TDistTime", ['customer_from','customer_to','distance','time'])
        self.TLatenessCost = namedtuple("TLatenessCost", ['customer','late_delivery_cost'])
        self.TOrderDetails = namedtuple("TOrderDetails", ['customer','earliest_delivery_time','latest_delivery_time','volume'])
        self.TDriver = namedtuple("TDriver", ['driver','startlocation_starttime','start_location','end_location','endlocation_starttime','route_assigned','travel_time',
                                              'wait_time','late_time','serve_time','current_load'])
        self.TDepot = namedtuple("TDepot", ['time','volume_remaining'])
        self.TDeliveries = namedtuple("TDeliveries", ['customer_from','customer_to','driver'])
        self.TDriverTruck = namedtuple("TDriverTruck", ['driver','truck','capacity','truck_assigned'])
        
        
    def date_to_num(self, dt):
        "Convert date to numeric"
        return int((dt - self.model.anchor_date).total_seconds() / 3600)

    def num_to_datetime(self, hour):
        "Convert hour(int) to datetime based on anchor date"
        beginning_date = self.model.anchor_date
        i = math.floor(hour / 24)
        j = hour - 24 * i
        date = beginning_date + timedelta(days=i)
        date_time = date + timedelta(hours=j)
        return date_time
    
    def convert_dt_to_num(self):
        def _convert_time_to_num(df: pd.DataFrame, col):
            "Applying date2num to an entire column"
            df.loc[:, col] = df[col].map(self.date_to_num)
            
        _convert_time_to_num(
            self.model.df_order, 'EarliestDeliveryTime')
        _convert_time_to_num(
            self.model.df_order, 'LatestDeliveryTime')
        _convert_time_to_num(
            self.model.df_roster, 'Start_time')
        _convert_time_to_num(
            self.model.df_roster, 'End_time')
        
        
    def create_df_depot(self):
        depot_list = []
        for i in range(60*24):
            depot_list.append((i,self.model.continuous_var(lb=0)))
        df_depot = pd.DataFrame(depot_list,columns=['time','volume'])
        return df_depot
    
    def create_df_driver(self):
        driver_deliveries = []
        for driver in self.model.driver_list:
            for cust_from in self.model.unique_cust_list:
                for cust_to in self.model.unique_cust_list:
                    if cust_from != cust_to:
                        serve_time = self.model.get_order_volume[cust_to]/self.model.unload_rate
                        travel_time = self.model.get_dist_time[(cust_from,cust_to)]
                        startlocation_starttime = self.model.continuous_var(lb=0)
                        endlocation_starttime = self.model.continuous_var(lb=0)
                        route_assigned = self.model.binary_var()
                        wait_time = self.model.continuous_var()
                        late_time = self.model.continuous_var()
                        current_load = self.model.continuous_var()
                        driver_deliveries.append((driver,startlocation_starttime,cust_from,cust_to,endlocation_starttime,route_assigned,travel_time,wait_time,serve_time,current_load))
        df_driver_deliveries = pd.DataFrame(driver_deliveries,columns=['driver','startlocation_starttime','start_location','end_location','endlocation_starttime','assigned',
                                                                       'travel_time','wait_time','late_time','serve_time','current_load'])
        return df_driver_deliveries
    
    
    def create_truck_driver(self):
        driver_truck_assign_list = []
        for _,row in self.model.driver_truck_assign.iterrows():
            driver_truck_assign_list.append((row.driver_id,row.Truck,row.Capacity,self.model.binary_var()))
        df_driver_truck_assign = pd.DataFrame(driver_truck_assign_list,columns=['driver','truck','capacity','assigned'])
        
        return
                
                                
                
            
    def load_model_objects(self):
        self.model.wait_time_cost = self.model.df_wait_cost['DriverWaitTimeCost'][0]
        self.model.anchor_date = self.model.df_roster.Start_time.min()
        self.convert_dt_to_num()
        
        #Loading data into namedtuples
        self.model.dist_time = [self.TDistTime(*row) for _,row in self.model.df_dist_time.iterrows()]
        self.model.get_dist_time = {(t.customer_from,t.customer_to):t.time for t in self.model.dist_time}
        self.model.order = [self.TOrderDetails(*row) for _,row in self.model.df_order.iterrows()]
        self.model.get_order_volume = {t.customer:t.volume for t in self.model.order}
        self.model.late_cost = [self.TLatenessCost(*row) for _,row in self.model.df_lateness_cost.iterrows()]
        
        #Object for depot
        df_depot = self.create_df_depot()
        self.model.depot = [self.TDepot(*row) for _,row in df_depot.iterrows()]
        
        # Object for driver-deliveries
        self.model.unique_cust_list = list(self.model.df_dist_time.cus_to.unique())
        self.model.unique_cust_list.remove('DEPOT01')
        self.model.driver_list = list(self.model.df_driver.driver_id.unique())
        df_driver_deliveries = self.create_df_driver()
        self.model.deliveries = [self.TDriver(*row) for _,row in df_driver_deliveries.iterrows()]
        
        
        
        #Object for truck-driver assignment
        self.model.driver_truck_assign = pd.merge(self.model.df_driver,self.model.df_truck_size,left_on='qualification',
                                                  right_on='Truck').drop('qualification',axis=1)
        df_truck_driver = self.create_truck_driver()
        
    
class SetupData(ModelObjects):

    def __init__(self, data, model: Model):
        ModelObjects.__init__(self, data, model)
        self.model.VAR = []
        self.model.periods = {}
        self.validation_issues = {}

        self.create_namedtuples()
        self.read_input_data_excel()
        self.load_input_data_excel()
        # self.validate_data()
        self.load_model_objects()
        # self.init_model_parameters()

class SetupConstraints(ModelObjects):

    def __init__(self, data, model: Model):
        self.input_json = data
        self.model = model
        
    
        
    def assignment_constraint(self):
        for cust in self.model.unique_cust_list:
            assign_list_cust_from = [v.route_assigned for v in self.model.deliveries if v.start_location == cust]
            self.model.add_constraint(self.model.sum(v for v in assign_list_cust_from) == 1)
            assign_list_cust_to = [v.route_assigned for v in self.model.deliveries if v.end_location == cust]
            self.model.add_constraint(self.model.sum(v for v in assign_list_cust_to) == 1)
            
        for driver in self.model.driver_list:
            assign_list_driver = [v.route_assigned for v in self.model.deliveries if v.driver == driver]
            self.model.add_constraint(self.model.sum(v for v in assign_list_driver) == 1)            
       
        return
    
    def start_wait_time_constraints(self):
        df_driver_deliveries = pd.DataFrame(self.model.deliveries)
        for _,row in df_driver_deliveries.iterrows():
            driver = row.driver
            cust_from = row.start_location
            cust_to = row.end_location
            SL_starttime = row.startlocation_starttime
            EL_starttime = row.endlocation_starttime
            travel_time = row.travel_time
            assigned_var = row.route_assigned
            wait_time = row.wait_time
            serve_time = row.serve_time
            late_time = row.late_time
            M = 40000
            
            #Wait time and start time constraints for start and end location
            self.model.add_constraint(SL_starttime + serve_time + travel_time - (1-assigned_var)*M + wait_time <= EL_starttime)
            
            # Contraint to restrict start time of customer after the earliest possible start time
            earliest_start_time = [i.earliest_delivery_time for i in self.model.order if i.customer == cust_to][0]
            self.model.add_constraint(EL_starttime >= earliest_start_time)
            
            # Constraint to assign late time based on latest delivery time
            latest_start_time = [i.latest_delivery_time for i in self.model.order if i.customer == cust_to][0]
            self.model.add_contraint(EL_starttime - late_time <= latest_start_time)
            
            # Constraint to restrict start times, wait_time and late_time to 0 if route is not assigned to driver
            self.model.add_constraint(SL_starttime <= assigned_var)
            self.model.add_constraint(SL_starttime <= assigned_var)
            self.model.add_constraint(SL_starttime <= assigned_var)
            self.model.add_constraint(SL_starttime <= assigned_var)
            
            
            
        return 
        
        
    

        
class ModelBuild:

    def __init__(self, data, scenario_id=None):
        self.model = Model()
        self.model.start_time = time.time()
        self.data = data
        self.scenario_id = scenario_id

    def create_model_run(self):
        "Creating model run"
        self.Setup_Data = SetupData(self.data, self.model)
        # self.Setup_Data.setup_data()
        # print("Done setup data")
        # self.Setup_Constraints = SetupConstraints(self.data, self.model)
        # self.Setup_Constraints.add_constraints()
        # self.Setup_Objectives = SetupObjectives(self.data, self.model)
        # self.Setup_Objectives.setup_objectives()
        # self.Model_Solve = ModelSolve(self.data, self.model)
        # self.solution = self.Model_Solve.solve_model()
        # self.Create_Results = CreateResults(
        #     self.data, self.model, self.solution)
        # self.Create_Results.create_results()
        
        
if __name__ == "__main__":


    mb = ModelBuild('Routing_input_new.xlsx')
    mb.create_model_run()
    # send_high_level_result()
    # send_detail_result()
        