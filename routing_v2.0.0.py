# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 09:21:11 2022

@author: Bronwyn
"""

from collections import namedtuple
import datetime
from lib2to3.pgen2 import driver
from pydoc import doc
from sqlite3 import Timestamp
from typing import List
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


# TWaitCost = namedtuple("TWaitCost", ['wait_cost'])
TDistTime = namedtuple("TDistTime", ['customer_from','customer_to','distance','time'])
TTruck = namedtuple("TTruck", ['type','capacity','quantity'])
TRoster = namedtuple("TRoster", ['driver','start_time','end_time'])
TLatenessCost = namedtuple("TLatenessCost", ['customer','late_delivery_cost'])
TOrderDetail = namedtuple("TOrderDetails", ['customer','earliest_delivery_time','latest_delivery_time','volume'])

TDelivery = namedtuple("TDelivery", ['driver_id', 'current_cus', 'previous_cus', 'assigned_var','arrive_time_var','remained_volume_var','capacity_truck']) # , ''
        
class ModelObjects:
    def __init__(self, data, model: Model):
        self._data = data
        self.cplex = model
        
        
    def is_overlap(self, a, b, r=0):
        """
        a: tuple of range, ex (1,5)
        b: tuple of range, ex (4,10)
        r: overlap size, ex 1
        return True
        """
        return a[0] < b[1]+r and b[0] < a[1]+r
    
    def load_input_data_excel(self):
        xlsx = pd.ExcelFile(self._data)
        self.df_wait_cost = pd.read_excel(xlsx,'DriverWaitTimeCost')
        self.df_dist_time = pd.read_excel(xlsx,'Distance_time')
        self.df_lateness_cost = pd.read_excel(xlsx,'UnitCostLateness')
        self.df_order = pd.read_excel(xlsx,'OrderDetails')
        self.df_driver = pd.read_excel(xlsx,'Driver')
        self.df_truck_size = pd.read_excel(xlsx,'Truck size')
        self.df_roster = pd.read_excel(xlsx,'Roster_Profile')
        
    def date_to_num(self, dt:datetime.datetime):
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
                
    def load_model_objects(self):
        self.vehicle_speed = 50 # km/h 
        self.unload_rate = 100 # units per hours
        
        self.wait_time_cost = self.df_wait_cost['DriverWaitTimeCost'][0]
        self.anchor_date:datetime.datetime = self.df_roster.Start_time.min()
        self._convert_dt_to_num()
        
        #Loading data into namedtuples
        self.dist_time = [TDistTime(*row) for _,row in self.df_dist_time.iterrows()]
        self.get_dist_time = {(t.customer_from,t.customer_to):t.time for t in self.dist_time}
        self.orders_list = [TOrderDetail(*row) for _,row in self.df_order.iterrows()]
        self.get_order_volume = {t.customer:t.volume for t in self.orders_list}
        self.late_cost = [TLatenessCost(*row) for _,row in self.df_lateness_cost.iterrows()]
        
        # Object for driver-deliveries
        self.unique_cust_list = list(self.df_dist_time.cus_to.unique())
        self.depot_key='DEPOT01'
        self.unique_cust_list.remove(self.depot_key)
        self.driver_list = list(self.df_driver.driver_id.unique())
        
        #Object for truck-driver assignment
        self.driver_truck_assign = pd.merge(self.df_driver,self.df_truck_size,left_on='qualification',
                                                  right_on='Truck').drop('qualification',axis=1)
        
        self.roster_list = list(TRoster(*row) for _,row in self.df_roster.iterrows())
        self.truck_list = list(TTruck(*row) for _,row in self.df_truck_size.iterrows())
        self.truck_get = defaultdict(lambda:None,{t.type:t for t in self.truck_list})


    def get_orders_by_range(self, start_time, end_time):
        a = (start_time, end_time)
        return [order 
                for order in self.orders_list 
                if self.is_overlap(a, (order.earliest_delivery_time, order.latest_delivery_time))]
        
    def get_time(self,cusfrom,custo):
        print([d.time for d in self.dist_time if d.customer_from == cusfrom and d.customer_to == custo])
        return int([d.time for d in self.dist_time if d.customer_from == cusfrom and d.customer_to == custo][0]) 
    def get_distance(self,cusfrom,custo ):
        return float([d.distance for d in self.dist_time if d.customer_from == cusfrom and d.customer_to == custo][0])
        
    def create_routes(self):
        self.ALL_DELIVERIES = []
        self.deliveries_by_roster:defaultdict[TOrderDetail,list[TDelivery]] = defaultdict(lambda: list())
        self.deliveries_by_order:defaultdict[TOrderDetail,list[TDelivery]] = defaultdict(lambda: list())
        self.t_depot = TOrderDetail(self.depot_key,0,0,0)
        
        for roster in self.roster_list:
            get_truck = self.driver_truck_assign[self.driver_truck_assign['driver_id']==roster.driver]
          
            truck_capacity = int(get_truck.Capacity)
            driver_id = get_truck.driver_id
            self.deliveries_by_roster[roster] = []
            order_list = self.get_orders_by_range(roster.start_time, roster.end_time) 

            
            order_list = order_list + [self.t_depot]
            for order_previous in order_list: 
                for order_current in order_list:
                    
                    if order_previous==order_current:
                        continue
                    
                    t=TDelivery(
                        assigned_var=self.cplex.binary_var(),
                        arrive_time_var=self.cplex.continuous_var(),
                        current_cus=order_current,
                        remained_volume_var=self.cplex.continuous_var(),
                        driver_id=driver_id,
                        previous_cus=order_previous,
                        capacity_truck = truck_capacity
                        
                    )   
                    
                    self.ALL_DELIVERIES.append(t)
                    self.deliveries_by_roster[roster].append(t)
                    self.deliveries_by_order[order_current].append(t)
                    self.deliveries_by_order[order_previous].append(t)
        

  


           
class SetupData():
    def __init__(self, model: ModelObjects):
        self.model = model
        self.validation_issues = {}
    
        self.model.load_input_data_excel()
        # self.validate_data()
        self.model.load_model_objects()
        self.model.create_routes()



class SetupConstraints():
    def __init__(self, model: ModelObjects):
        self.model = model
        self.cplex=self.model.cplex

    def add_order_constraint(self):
        "Creating constraints"
        for order in self.model.orders_list:
            
            list_delivering = self.model.deliveries_by_order[order]

            get_outbound = [d.assigned_var for d in list_delivering if d.current_cus==order.customer]
            get_inbound = [d.assigned_var for d in list_delivering if d.previous_cus==order.customer]
            
            if len(get_inbound)>0:
                self.cplex.add_constraint(self.cplex.sum(get_inbound)==1)
            if len(get_outbound)>0:
                self.cplex.add_constraint(self.cplex.sum(get_outbound)==1)
    def add_depot_constraints(self):
        "Creating constraints related to depot"
        list_deliveries_depot =  self.model.deliveries_by_order[self.model.t_depot]
        get_outbound = [d.assigned_var for d in list_deliveries_depot if d.current_cus==self.model.t_depot]
        get_inbound = [d.assigned_var for d in list_deliveries_depot if d.previous_cus==self.model.t_depot]
        self.cplex.add_constraint(self.cplex.sum(get_inbound) ==  sum(get_outbound))

    def add_time_constraints(self): 
        "Creating time constraints"
        for order,list_delivering in self.model.deliveries_by_order.items():
            for d in list_delivering: 
                waiting_time = self.cplex.max(0,order.earliest_delivery_time-d.arrive_time_var)
                self.cplex.add_constraint(d.arrive_time_var + waiting_time >= order.earliest_delivery_time)
                self.cplex.add_constraint(d.arrive_time_var + waiting_time <= order.latest_delivery_time)
                previous_cus = d.previous_cus
                t_j = d.arrive_time_var 
                get_inbounds_of_previous = [_d for _d in list_delivering if _d.current_cus==previous_cus]
                for pre_cus in get_inbounds_of_previous:
                    t_i = pre_cus.arrive_time_var 
                    ind_exist_both = self.cplex.binary_var()
                    self.cplex.add_constraint(ind_exist_both == d.assigned_var * pre_cus.assigned_var)
                    self.cplex.add_indicator(ind_exist_both , t_i + waiting_time + self.model.get_time(pre_cus.current_cus,d.current_cus))


        
    def add_vehilce_loading_constraints(self):
        for order,list_delivering in tqdm(self.model.deliveries_by_order.items()):
            
            for d in list_delivering: 
                if d.previous_cus == self.model.t_depot: 
                    
                    self.cplex.add_constraint(d.remained_volume_var + order.volume <= d.capacity_truck)

                    
                else:    
                    # mình là d
                    previous_cus = d.previous_cus 
                    Z_j = d.remained_volume_var # load hiện tại
                    self.cplex.add_constraint(d.remained_volume_var <= d.capacity_truck)
                    get_inbounds_of_previous = [_d for _d in list_delivering if _d.current_cus==previous_cus]
                    
                    for pre_cus in get_inbounds_of_previous:
                       
                        Z_i = pre_cus.remained_volume_var
                        ind_exist_both = self.cplex.binary_var()
                        self.cplex.add_constraint(ind_exist_both == d.assigned_var * pre_cus.assigned_var)
                        # load hiện tại = load trước đó - load-đã dỡ xuống ở chỗ hiện tại
                        self.cplex.add_indicator(ind_exist_both , Z_j + order.volume == Z_i) 

        

            
    def add_constraints(self):
        self.add_order_constraint()
        self.add_depot_constraints()
        self.add_time_constraints()
        self.add_vehilce_loading_constraints()
        

class SetupObjectives():
    def __init__(self,model:  ModelObjects):
        self.model = model
        self.cplex=self.model.cplex
    
    def add_objectives(self):
        ''
        self.model.total_distance = 0 
        self.model.total_distance = self.cplex.sum((s.assigned_var*self.model.get_time(s.previous_cus,s.current_cus) for s in self.model.deliveries_by_order))
        





        # add  kpi 
        self.model.add_kpi(self.model.total_distance, "Total distance") 


        # add objective 
        self.model.minimize_static_lex([self.model.total_distance])
class ModelSolve: 
    def __init__(self, model: ModelObjects):
        self.model = model
        self.cplex=self.model.cplex
    def solve_model(self): 
        ''

class ModelBuild:

    def __init__(self, data, scenario_id=None):
        self.start_time = time.time()
        self.data = data
        self.scenario_id = scenario_id
        docplex = Model()
        self.model = ModelObjects(self.data,docplex)
        
    def create_model_run(self):
        "Creating model run"
        self.Setup_Data = SetupData(self.model)
        # self.Setup_Data.setup_data()
        print("Done setup data")
        self.Setup_Constraints = SetupConstraints(self.model)
        self.Setup_Constraints.add_constraints()
        self.Setup_Objectives = SetupObjectives(self.model)
        self.Setup_Objectives.add_objectives()
        self.Model_Solve = ModelSolve(self.data, self.model)
        # self.solution = self.Model_Solve.solve_model()
        # self.Create_Results = CreateResults(
        #     self.data, self.model, self.solution)
        # self.Create_Results.create_results()
        
        
if __name__ == "__main__":


    mb = ModelBuild('data_route.xlsx')
    mb.create_model_run()
    # send_high_level_result()
    # send_detail_result()
        