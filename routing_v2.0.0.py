# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 09:21:11 2022

@author: Bronwyn
"""

from collections import namedtuple
import datetime
from dis import dis
from lib2to3.pgen2 import driver
from multiprocessing.connection import wait
from pydoc import doc
from random import randrange
from sqlite3 import Timestamp
from typing import List
from tqdm import tqdm
import pandas as pd
from docplex.mp.model import Model
from docplex.util.environment import get_environment
from functools import reduce, total_ordering
import numpy as np
import uuid
import math
from datetime import date, timedelta
import sys
import time
import json
from pandas import json_normalize
import requests
import os
from collections import defaultdict
from docplex.mp.solution import SolveSolution
from docplex.mp.utils import DOcplexException



# TWaitCost = namedtuple("TWaitCost", ['wait_cost'])
TDistTime = namedtuple("TDistTime", ['customer_from','customer_to','distance','time'])
TTruck = namedtuple("TTruck", ['type','capacity','quantity'])
TRoster = namedtuple("TRoster", ['driver','start_time','end_time'])
TLatenessCost = namedtuple("TLatenessCost", ['customer','late_delivery_cost'])
TOrderDetail = namedtuple("TOrderDetails", ['customer','earliest_delivery_time','latest_delivery_time','volume'])

TDelivery = namedtuple("TDelivery", ['driver_id', 'current_cus', 'previous_cus', 'assigned_var','arrive_time_var','waiting_time_var','remained_volume_var','capacity_truck']) # , ''
        
class ModelObject:
    def __init__(self, model: Model):
        self.cplex = model
        
        ##
        self.total_wating_time = 0
        self.slack_orders = []
        self.late_time_by_delivery = defaultdict(lambda: 0)
        
    def is_overlap(self, a, b, r=0):
        """
        a: tuple of range, ex (1,5)
        b: tuple of range, ex (4,10)
        r: overlap size, ex 1
        return True
        """
      
        return a[0] < b[1]+r and b[0] < a[1]+r
    
    def load_input_data_excel(self,input_file):
        xlsx = pd.ExcelFile(input_file)
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
        self.orders_list = self.orders_list[:]
        self.get_order_volume = {t.customer:t.volume for t in self.orders_list}
        self.late_cost = [TLatenessCost(*row) for _,row in self.df_lateness_cost.iterrows()]
        
        # Object for driver-deliveries
        self.unique_cust_list = list(self.df_dist_time.cus_to.unique())
        self.depot_key='DEPOT01'
        self.unique_cust_list.remove(self.depot_key)
        self.driver_list = list(self.df_driver.driver_id.unique())[:]
        
        #Object for truck-driver assignment
        self.driver_truck_assign = pd.merge(self.df_driver,self.df_truck_size,left_on='qualification',
                                                  right_on='Truck').drop('qualification',axis=1)
        
        self.roster_list = list(TRoster(*row) for _,row in self.df_roster.iterrows())[:]
        self.truck_list = list(TTruck(*row) for _,row in self.df_truck_size.iterrows())
        self.truck_get = defaultdict(lambda:None,{t.type:t for t in self.truck_list})
        

    def get_orders_by_range(self, start_time, end_time):
        a = (start_time, end_time)
        return [order 
                for order in self.orders_list 
                if self.is_overlap(a, (order.earliest_delivery_time, order.latest_delivery_time))]
        
    def get_time(self,cusfrom,custo):
        try:
            return int([d.time for d in self.dist_time if d.customer_from == cusfrom and d.customer_to == custo][0])/60
        except:
            print("error get_time: ", cusfrom,custo)
    def get_distance(self,cusfrom,custo ):
        try:
            return float([d.distance for d in self.dist_time if d.customer_from == cusfrom and d.customer_to == custo][0])
        except:
            print("error get_distance: ", cusfrom,custo)

    def create_routes(self):
        self.ALL_DELIVERIES:List[TDelivery] = []
        self.deliveries_by_roster:defaultdict[TOrderDetail,list[TDelivery]] = defaultdict(lambda: list())
        self.inbound_deliveries_by_order:defaultdict[str,list[TDelivery]] = defaultdict(lambda: list())
        self.outbound_deliveries_by_order:defaultdict[str,list[TDelivery]] = defaultdict(lambda: list())
        self.deliveries_by_driver:defaultdict[str,list[TDelivery]] = defaultdict(lambda: list())
        waiting_times=np.arange(0,5,0.15)
        # ----------------------------------------------------------------------------------
        self.depot = TOrderDetail(self.depot_key,0,0,0)
        for roster in self.roster_list:
            get_truck = self.driver_truck_assign.loc[self.driver_truck_assign['driver_id']==roster.driver,:]
            truck_capacity = int(list(get_truck['Capacity'])[0])
            driver_id = list(get_truck['driver_id'])[0]
            order_list = self.get_orders_by_range(roster.start_time, roster.end_time) 
            order_list = order_list + [self.depot]
            for order_current in order_list:
                lst = []
                for order_previous in order_list: 
                    if order_previous==order_current:
                        continue
                    if order_current.customer != self.depot_key and order_previous.customer != self.depot_key and \
                        order_previous.latest_delivery_time > order_current.latest_delivery_time and \
                            order_current.latest_delivery_time - order_previous.earliest_delivery_time < self.get_time(order_previous.customer,order_current.customer):
                                # current customer |======================>
                                # previous customer             |========================>
                                # moving time                   |--------->
                                continue
                    
                    t = TDelivery(
                        assigned_var=self.cplex.binary_var(f"D_{driver_id}_{order_previous.customer}_{order_current.customer}"),
                        arrive_time_var=self.cplex.integer_var(lb=round(roster.start_time*60),ub=round(roster.end_time*60)),
                        waiting_time_var=self.cplex.integer_var(lb=0,ub=5*12), # every 5 minute
                        current_cus=order_current.customer,
                        remained_volume_var=self.cplex.integer_var(lb=0,ub=truck_capacity),
                        driver_id=driver_id,
                        previous_cus=order_previous.customer,
                        capacity_truck = truck_capacity
                    ) 
                    # self.cplex.add_constraint(1==self.cplex.sum((a==t.arrive_time_var) for a in np.arange(roster.start_time,roster.end_time,0.15)))
                    # self.cplex.add_constraint(1==self.cplex.sum((a==t.waiting_time_var) for a in waiting_times))
                    
                    lst.append((order_previous, order_current, t))
                
                # Filtering 25% of the longest delivery
                if len(lst)>20:
                    lst.sort(key=lambda x: self.get_distance(x[0].customer,x[1].customer))
                    lst = lst[0:math.floor(0.40*len(lst))]
                    
                for pr,cu,t in lst:
                    self.ALL_DELIVERIES.append(t)
                    self.deliveries_by_roster[roster].append(t)
                    self.inbound_deliveries_by_order[cu.customer].append(t)
                    self.outbound_deliveries_by_order[pr.customer].append(t)
                    self.deliveries_by_driver[driver_id].append(t)
                    
        self.order_get = defaultdict(lambda:None,{o.customer:o for o in self.orders_list+[self.depot]})
        print("---------------------------------",len(self.ALL_DELIVERIES))
        
    
class SetupData():
    def __init__(self, model: ModelObject):
        self.model = model
        
        #------------------------------
        self.validation_issues = {}
        
    def setup_data(self,input_file):
        self.model.load_input_data_excel(input_file)
      
        self.model.load_model_objects()
        self.model.create_routes()


class SetupConstraints():
    def __init__(self, model: ModelObject):
        self.model = model
        self.cplex=self.model.cplex

    def add_order_constraint(self):
        "Creating constraints"
        # CONSTRAINT: driver are not allowed to return a customer after leaving it
        for driver in self.model.deliveries_by_driver:
            for i in self.model.deliveries_by_driver[driver]:
                for j in self.model.deliveries_by_driver[driver]:
                    if (i.current_cus!=self.model.depot_key or i.previous_cus!=self.model.depot_key) \
                        and i.previous_cus == j.current_cus and i.current_cus == j.previous_cus: 
                            list_cur_pre = [i.assigned_var,j.assigned_var]
                            self.cplex.add_constraint(sum(list_cur_pre) <=1)
                
        for order in tqdm(self.model.orders_list):
            get_outbound = [d.assigned_var for d in self.model.outbound_deliveries_by_order[order.customer]]
            get_inbound = [d.assigned_var for d in self.model.inbound_deliveries_by_order[order.customer]]

            id =  len(self.model.slack_orders)
            slack = self.cplex.binary_var(name=f"slack_{order.customer}")
            self.model.slack_orders.append(slack)
            
            sum_inbound = self.cplex.sum(get_inbound)
            sum_outbound = self.cplex.sum(get_outbound)
            
            if len(get_inbound)==0 or len(get_outbound)==0:
                self.cplex.add_constraint(slack==1)
                continue
            
            # CONSTRAINT: total inbound and total outbound of an order must be equal to 1
            # in case this order can not be delivered (slack=1), then total inbound == outbound == 0
            self.cplex.add_indicator(slack, sum_outbound==0, 1)
            self.cplex.add_indicator(slack, sum_inbound==0, 1)
            self.cplex.add_indicator(slack, sum_outbound==1, 0)
            self.cplex.add_indicator(slack, sum_inbound==1, 0)
            
            
            # Group all deliveries of this order by driver_id
            group_by_driver = defaultdict(lambda: list())
            for d in self.model.outbound_deliveries_by_order[order.customer]:
                group_by_driver[d.driver_id].append(d.assigned_var)
            for d in self.model.inbound_deliveries_by_order[order.customer]:
                group_by_driver[d.driver_id].append(d.assigned_var)
            
            # CONSTRAINT: only one driver can take this order
            self.cplex.add_indicator(slack, self.cplex.sum(self.cplex.logical_or(*_l) for _,_l in group_by_driver.items())==1, 0)
            self.cplex.add_indicator(slack, self.cplex.sum(self.cplex.logical_or(*_l) for _,_l in group_by_driver.items())==0, 1)
        
    
    def add_depot_constraints(self):
        "Creating constraints related to depot"
        for driver,_ in tqdm(self.model.deliveries_by_driver.items()):
            k = self.model.depot.customer
            get_outbound = [d.assigned_var for d in self.model.outbound_deliveries_by_order[k] if d.driver_id==driver]
            get_inbound = [d.assigned_var for d in self.model.inbound_deliveries_by_order[k] if d.driver_id==driver]
            # CONSTRAINT: for each driver, the total times he leaves the depot equals to the total times he goes back
            self.cplex.add_constraint(self.cplex.sum(get_inbound) ==  self.cplex.sum(get_outbound))
        
        dict_isNotTheHeadVars_groupBy_driver = defaultdict(lambda: [])
        connectedPathVar_list = []

        for order in tqdm(self.model.orders_list+[self.model.depot]):
            for d in self.model.inbound_deliveries_by_order[order.customer]: 
                if d.previous_cus == self.model.depot_key:
                    moving_time = self.model.get_time(d.previous_cus,d.current_cus)
                    get_pre_inbounds_of_driver = [ib for ib in self.model.inbound_deliveries_by_order[d.previous_cus] if ib.driver_id==d.driver_id]
                    for pre_delivery in get_pre_inbounds_of_driver:
                        is_not_theHead_var = self.cplex.binary_var(name=f"is_not_the_head[{d.driver_id}:{pre_delivery.previous_cus}->{d.previous_cus}->{d.current_cus}]")
                        is_connected_var = self.cplex.binary_var(name=f"is_connected[{d.driver_id}:{pre_delivery.previous_cus}->{d.previous_cus}->{d.current_cus}]")
                        
                        # self.cplex.add_if_then(is_not_theHead_var==0, pre_delivery.assigned_var+d.assigned_var==2)
                        # self.cplex.add_if_then(is_not_theHead_var==0, is_connected_var==0)
                        self.cplex.add_if_then(is_connected_var==1, pre_delivery.assigned_var+d.assigned_var==2)
                        
                        dict_isNotTheHeadVars_groupBy_driver[d.driver_id].append(is_not_theHead_var)
                        connectedPathVar_list.append((pre_delivery,d,is_not_theHead_var,is_connected_var))
                        is_existPath = self.cplex.logical_and(d.assigned_var,pre_delivery.assigned_var,is_not_theHead_var,is_connected_var)
                        self.cplex.add_if_then(is_existPath==1,pre_delivery.arrive_time_var/60 + pre_delivery.waiting_time_var/12 + moving_time == d.arrive_time_var/60)
                    
                    
                    # # CONSTRAINT: considering a not-the-head delivery starting from depot (d[0->j]), there is one and only one back-to-depot delivery (d[i->0]) connect to it
                    # # for example: 
                    # #   + At the the depot, let's consider this 3 exist deliveres: d01=d[0->1], d20=d[2->0], d30=d[3->0], can be formed to 2 cut-paths p1={d20 d01}, p2={d30 d01}
                    # #   => So, neither p1 or p2 are connected 
                    # if len(list_isConnectedVars)>0:
                    #     self.cplex.add_if_then(self.cplex.sum(self.cplex.logical_and(x[0],x[1],x[2]) for x in list_isConnectedVars)>=1,self.cplex.sum(x[3] for x in list_isConnectedVars)==1)
        df_connectedPathVar = pd.DataFrame(columns=["pre","cur","isNotHead","isConnected"],data=connectedPathVar_list)
        for _,group in df_connectedPathVar.groupby(by=["pre"]):
            sum1 = 0
            sum2 = 0
            for t in group.itertuples():
                sum1 += self.cplex.logical_and(t.pre.assigned_var,t.cur.assigned_var,t.isNotHead)
                sum2 += t.isConnected
                
            self.cplex.add_if_then(sum1>=1,sum2==1)
        
        for _,group in df_connectedPathVar.groupby(by=["cur"]):
            sum1 = 0
            sum2 = 0
            for t in group.itertuples():
                sum1 += self.cplex.logical_and(t.pre.assigned_var,t.cur.assigned_var,t.isNotHead)
                sum2 += t.isConnected
                
            self.cplex.add_if_then(sum1>=1,sum2==1)
        self.model.xxx = df_connectedPathVar
        for driver_id,list_isNotTheHeadVars in dict_isNotTheHeadVars_groupBy_driver.items():
            # CONSTRAINT: for each driver, there is only one delivery that can be the head (starting) of his route
            if len(list_isNotTheHeadVars)>0:
                self.cplex.add_constraint(self.cplex.if_then(self.cplex.sum(d.assigned_var for d in self.model.deliveries_by_driver[driver_id])>=1,self.cplex.sum(self.cplex.logical_not(v) for v in list_isNotTheHeadVars)==1),ctname=f"only one delivery can be the head {driver_id}")

        #             else:
        #                 ""
        #                 # ELSE, the Var not only depends on the existance of the 1st node and 3nd node but also depends on 2 more requirements bellow
        #                 # First, for each driver, there is only one delivery that can be the head (starting) of his route
        #                 # delivery : d[i->j]
        #                 # d[0->1] (the Head) --> d[1->2] --> d[2->3] --> d[3->0] --> d[0->4] (not the head) --> d[4->0] (ending)
        #                 # Second, considering a not-the-head delivery starting from depot (d[0->j]), there is one and only one back-to-depot delivery (d[i->0]) connect to it
        #                 is_not_theHead_var = self.cplex.binary_var(name=f"is_not_the_head[{d.driver_id}:{pre_delivery.previous_cus}->{d.previous_cus}->{d.current_cus}]")
        #                 is_connected_var = self.cplex.binary_var(name=f"is_connected[{d.driver_id}:{pre_delivery.previous_cus}->{d.previous_cus}->{d.current_cus}]")
        #                 self.model.list_is_connected_var.append(is_connected_var)
                        
        #                 # self.cplex.add_if_then(is_connected_var==1,d.assigned_var+pre_delivery.assigned_var==2)
        #                 # self.cplex.add_if_then(is_not_theHead_var==0,d.assigned_var+pre_delivery.assigned_var==2)
        #                 self.cplex.add_if_then(is_connected_var==1,d.assigned_var+pre_delivery.assigned_var==2)
        #                 # self.cplex.add_if_then(is_not_theHead_var==0,is_connected_var==0)
        #                 dict_isNotTheHeadVars_groupBy_driver[d.driver_id].append(is_not_theHead_var)
        #                 list_isConnectedVars.append((d.assigned_var, pre_delivery.assigned_var,is_not_theHead_var, is_connected_var))
        #                 self.cplex.add_constraint(is_existPath_var == self.cplex.logical_and(d.assigned_var, pre_delivery.assigned_var,is_not_theHead_var,is_connected_var))


                    
        #         # CONSTRAINT: considering a not-the-head delivery starting from depot (d[0->j]), there is one and only one back-to-depot delivery (d[i->0]) connect to it
        #         # for example: 
        #         #   + At the the depot, let's consider this 3 exist deliveres: d01=d[0->1], d20=d[2->0], d30=d[3->0], can be formed to 2 cut-paths p1={d20 d01}, p2={d30 d01}
        #         #   => So, neither p1 or p2 are connected 
        #         if len(list_isConnectedVars)>0:
        #             self.cplex.add_if_then(self.cplex.sum(self.cplex.logical_and(x[0],x[1],x[2]) for x in list_isConnectedVars)>=1,self.cplex.sum(x[3] for x in list_isConnectedVars)==1)
        # #-------       

            
                  
            
    def add_time_constraints(self): 
        "Creating time constraints"

        #-------
        for order in tqdm(self.model.orders_list+[self.model.depot]):
            list_delivering = self.model.inbound_deliveries_by_order[order.customer]
            for d in list_delivering: 

                current_arriving_time = d.arrive_time_var/60
                
                if order!=self.model.depot:
                    waiting_time=d.waiting_time_var/12

                    late_time = self.cplex.max(0,current_arriving_time-order.latest_delivery_time)
                    self.model.late_time_by_delivery[d] = late_time
                    # CONSTRAINT: for waiting_time and late_time
                    self.cplex.add_indicator(d.assigned_var,current_arriving_time + waiting_time >= order.earliest_delivery_time)
                    self.cplex.add_indicator(d.assigned_var,current_arriving_time - late_time <= order.latest_delivery_time)
          
                moving_time = self.model.get_time(d.previous_cus,d.current_cus)

                get_pre_inbounds_of_driver = [ib for ib in self.model.inbound_deliveries_by_order[d.previous_cus] if ib.driver_id==d.driver_id]
                
                for pre_delivery in get_pre_inbounds_of_driver:
                    pre_arriving_time = pre_delivery.arrive_time_var/60
                    
                    # A cut-path: 1->2->3 is formed by 2 deliveries d[1->2] and d[2->3] and by 3 nodes (customers): 1,2,3 
                    pre_waiting_time=pre_delivery.waiting_time_var/12
                    if d.previous_cus != self.model.depot_key: 
                        # CONSTRAINT: if both d[1->2] and d[2->3] are exist then pre_arriving_time + pre_waiting_time + moving_time == current_arriving_time
                        self.cplex.add_if_then(pre_delivery.assigned_var+d.assigned_var==2, pre_arriving_time + pre_waiting_time + moving_time == current_arriving_time)
        
        
    def add_vehilce_loading_constraints(self):
        for order in tqdm(self.model.orders_list+[self.model.depot]):
            list_delivering = self.model.inbound_deliveries_by_order[order.customer]
            for d in list_delivering: 
                current_remain_volume = d.remained_volume_var
                
                #CONSTRAINT: the volume remaining after unloaded packages + volume of unloaded packages <= the truck's capacity
                self.cplex.add_indicator(d.assigned_var,d.remained_volume_var + order.volume <= d.capacity_truck)

                if d.previous_cus != self.model.depot_key:
                    get_order_inbounds_of_driver = [ib for ib in self.model.inbound_deliveries_by_order[d.previous_cus] if ib.driver_id==d.driver_id]
                    for pre_delivery in get_order_inbounds_of_driver:
                        pre_remain_volume = pre_delivery.remained_volume_var
                        # CONSTRAINT: the volume remaining after unloading packages + volume of unloaded packages == the volume remaining before arriving
                        self.cplex.add_if_then(pre_delivery.assigned_var+d.assigned_var==2 , current_remain_volume + order.volume == pre_remain_volume) 



    def add_constraints(self):
        self.add_order_constraint()
        self.add_depot_constraints()
        self.add_time_constraints()
        self.add_vehilce_loading_constraints()
        

class SetupObjectives():
    def __init__(self,model:  ModelObject):
        self.model = model
        self.cplex=self.model.cplex
    
    def add_objectives(self):
        ''
        distance_cost_rate = 0.5 # $/kms
        waiting_cost_rate = 15 # $/hours
        latency_cost_rate = 250 # $/hours
        latency_priority_rate = 10
        starting_cost_rate = 10 # $/truck_size
        
        delivery_costs = []
        for order in self.model.orders_list+[self.model.depot]:
            list_delivering = self.model.inbound_deliveries_by_order[order.customer]+self.model.outbound_deliveries_by_order[order.customer]
            for d in list_delivering:
                cost = 0
                
                dist_cost = self.model.get_distance(d.previous_cus,d.current_cus)*distance_cost_rate
                if d.previous_cus==self.model.depot.customer:
                    dist_cost += int(d.capacity_truck)*starting_cost_rate
                
                cost += dist_cost * d.assigned_var
                cost += d.waiting_time_var/12 * waiting_cost_rate
                cost += self.model.late_time_by_delivery[d] * latency_cost_rate * latency_priority_rate
                
                delivery_costs.append(cost)
            
                
        total_moving_cost = self.cplex.sum(delivery_costs)
        total_slack_orders = self.cplex.sum(self.model.slack_orders)
        total_wating_time = self.cplex.sum(d.waiting_time_var/12 for d in self.model.ALL_DELIVERIES)
        total_latency = self.cplex.sum(lt for lt in self.model.late_time_by_delivery.values())
        # add  kpi 
        # self.cplex.add_constraint(total_slack_orders <= len(self.model.slack_orders)/2,"num slacks") # Num of slack not higher than 50%
        self.cplex.add_kpi(total_moving_cost, "Total moving cost") 
        self.cplex.add_kpi(total_slack_orders, "Total slack orders") 
        self.cplex.add_kpi(total_wating_time, "Total waiting time") 
        self.cplex.add_kpi(total_latency, "Total latency time") 

        # add objectives
        self.cplex.minimize_static_lex([total_slack_orders,total_moving_cost])
        
class ModelSolve: 
    def __init__(self, model: ModelObject):
        self.model = model
        self.cplex=self.model.cplex
        
    def parameter_set(self, cplex, limit):
        ps = cplex.create_parameter_set()
        ps.add(cplex.parameters.timelimit, limit[0])
        ps.add(cplex.parameters.preprocessing.aggregator, limit[1])
        ps.add(cplex.parameters.mip.polishafter.solutions, limit[2])
        ps.add(cplex.parameters.barrier.algorithm, limit[3])
        ps.add(cplex.parameters.mip.tolerances.mipgap, 0.1)
        return ps

    def solve_model(self, solution_file):
        print("Solving the model...\n")
        if not self.cplex.has_multi_objective():
            raise Exception("Expecting a multi-objective model!")
        
        # Get the CPLEX instance from the docplex model
        self.cplex_ins = self.cplex.get_cplex()

        with open('cplex.log','w') as cplexlogs:
            self.cplex_ins.set_log_stream(cplexlogs)
            self.cplex_ins.set_results_stream(cplexlogs)
            self.cplex_ins.parameters.threads.set(10)
            # self.model.cplex_ins.runseeds(cnt=10)
            # Print the details of each multi-objective round
            self.cplex_ins.parameters.multiobjective.display.set(2)
            # model.cplex_ins.parameters.preprocessing.aggregator = 0
            # ps = [self.parameter_set_with_timelimit(self.model.cplex_ins, l) for l in [(937390,40,1,0),(1890885,20,1,3),(1865849,20,1,3)]]
            ps = [self.parameter_set(self.cplex_ins, params) for params in [(7200,40,5,1),(3600,40,10,1)]]
            solve = self.cplex_ins.solve(paramsets=ps)
            self.cplex_ins.solution.write(solution_file)            
            
            
class CreateResults():
    
    def __init__(self,model:ModelObject,solution:SolveSolution):
        self.model = model
        self.solution = solution
    def get_value(self,var):
        try:
            value = self.solution.get_value(var)
        except DOcplexException:
            value = var
            print("Solution get value failed: ", var, "(DOcplexException)")
        except:
            value = 0
            print("Solution get value failed: ", var, "(Unknown)")
            
        return value
    
    def create_results(self): 

        list_rows = []
        if self.solution is None:
            print("No solution found")
        
        if self.solution:
            
            for row in self.model.xxx.itertuples():
                if round(self.get_value(row.pre.assigned_var))+round(self.get_value(row.cur.assigned_var))==2:
                    print(row.pre.assigned_var.name,row.cur.assigned_var.name,int(self.get_value(row.isConnected)),("+"if round(self.get_value(row.isNotHead))==1 else "-----"))
            
            self.model.cplex.report_kpis(solution=self.solution)
            for d in self.model.ALL_DELIVERIES:
                if round(self.get_value(d.assigned_var))==1:
                    arrive_time = self.get_value(d.arrive_time_var)/60
                    waiting_time = self.get_value(d.waiting_time_var)/12
                    remain_volume = self.get_value(d.remained_volume_var)
                    latency_time = self.get_value(self.model.late_time_by_delivery[d])
                        
                    order = self.model.order_get[d.current_cus]
                    moving_time = self.model.get_time(d.previous_cus,d.current_cus)
                    list_rows.append((d.driver_id,d.previous_cus,d.current_cus,remain_volume,d.capacity_truck,arrive_time,waiting_time,order.earliest_delivery_time,order.latest_delivery_time,order.volume,moving_time,latency_time))
                    
            df = pd.DataFrame(columns=["Driver","Previous","Current","RemainVolume","Capacity","ArriveTime","WaitingTime","O.EarliestTime","O.LatestTime","O.Volume","MovingTime","Latency"],data=list_rows)
            df.sort_values(by=['Driver','ArriveTime'],inplace=True)
            df.to_excel("result.xlsx")
            print(df)
        
        
class ModelBuild:
    def __init__(self, input_file,solution_file):
        self.start_time = time.time()
        self.input_file = input_file
        self.solution_file = solution_file
        self.docplex = Model()
        self.model = ModelObject(self.docplex)
        
        
    def create_model_run(self):
        "Creating model run"
        self.Setup_Data = SetupData(self.model)
        self.Setup_Data.setup_data(self.input_file)
        print("Done setup data")
        
        self.Setup_Constraints = SetupConstraints(self.model)
        self.Setup_Constraints.add_constraints()
        print("Done setup constraints")
        
        self.Setup_Objectives = SetupObjectives(self.model)
        self.Setup_Objectives.add_objectives()
        print("Done setup objectives")
        
        self.Model_Solve = ModelSolve(self.model)
        load_solution = self.Model_Solve.solve_model(self.solution_file)
        print(f"Done solve the model >> saved >> {self.solution_file}")
        print("\n")

        load_solution = SolveSolution.from_file(self.solution_file, self.docplex)[0]
        print("Done load solution from file")
        
        self.Create_Results = CreateResults(self.model, load_solution)
        self.Create_Results.create_results()
        print("Done create results")
        
        
if __name__ == "__main__":

    _t0 = time.time()
    mb = ModelBuild('data_route.xlsx','model.sol')
    mb.create_model_run()
    print('================================================')
    print(time.time()-_t0)
    # send_high_level_result()
    # send_detail_result()
        