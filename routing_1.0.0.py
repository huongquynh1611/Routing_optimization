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
from docplex.mp.solution import SolveSolution

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
        self.TRoster = namedtuple("TRoster", ['driver','start_time','end_time'])
        
        
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
            driver_start = self.model.df_roster.loc[self.model.df_roster['Driver'] == driver]['Start_time'].values[0]
            driver_end = self.model.df_roster.loc[self.model.df_roster['Driver'] == driver]['End_time'].values[0]
            for cust_from in self.model.unique_cust_list:
                for cust_to in self.model.unique_cust_list:
                    earliest_st_time = [k.earliest_delivery_time for k in self.model.order if k.customer == cust_to][0]
                    latest_st_time = [k.latest_delivery_time for k in self.model.order if k.customer == cust_to][0]
                    if cust_from != cust_to and earliest_st_time > driver_start and latest_st_time < driver_end:
                        serve_time = self.model.get_order_volume[cust_from]/self.model.unload_rate
                        travel_time = self.model.get_dist_time[(cust_from,cust_to)]
                        startlocation_starttime = self.model.continuous_var(lb=0)
                        endlocation_starttime = self.model.continuous_var(lb=0)
                        route_assigned = self.model.binary_var()
                        wait_time = self.model.continuous_var(lb=0)
                        late_time = self.model.continuous_var(lb=0)
                        current_load = self.model.continuous_var(lb=0,ub=self.model.get_driver_truck[driver][1])

                        driver_deliveries.append((driver,startlocation_starttime,cust_from,cust_to,endlocation_starttime,route_assigned,travel_time,
                                                  wait_time,late_time,serve_time,current_load))
        df_driver_deliveries = pd.DataFrame(driver_deliveries,columns=['driver','startlocation_starttime','start_location','end_location','endlocation_starttime','assigned',
                                                                       'travel_time','wait_time','late_time','serve_time','current_load'])
        return df_driver_deliveries
    
    
    def create_truck_driver(self):
        driver_truck_assign_list = []
        for _,row in self.model.driver_truck_assign.iterrows():
            driver_truck_assign_list.append((row.driver_id,row.Truck,row.Capacity,self.model.binary_var()))
        df_driver_truck_assign = pd.DataFrame(driver_truck_assign_list,columns=['driver','truck','capacity','assigned'])
        
        return df_driver_truck_assign
            
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
        
        #Object for truck-driver assignment
        self.model.driver_truck_assign = pd.merge(self.model.df_driver,self.model.df_truck_size,left_on='qualification',
                                                  right_on='Truck').drop('qualification',axis=1)
        df_truck_driver = self.create_truck_driver()
        self.model.truck_driver_assign = [self.TDriverTruck(*row) for _,row in df_truck_driver.iterrows()]
        
        self.model.get_driver_truck = {t.driver:(t.truck,t.capacity,t.truck_assigned) for t in self.model.truck_driver_assign}
        
        # Object for driver-deliveries
        self.model.unique_cust_list = list(self.model.df_dist_time.cus_to.unique())
        # self.model.unique_cust_list.remove('DEPOT01')
        self.model.driver_list = list(self.model.df_driver.driver_id.unique())
        df_driver_deliveries = self.create_df_driver()
        self.model.deliveries = [self.TDriver(*row) for _,row in df_driver_deliveries.iterrows()]
        
        
        

    
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
        cust_list = [k for k in self.model.unique_cust_list if k != 'DEPOT01']
        self.model.no_delivery = []
        for cust in cust_list:
            
            # Constraint to restrict one path from a customer
            assign_list_cust_from = [v.route_assigned for v in self.model.deliveries if v.start_location == cust]
            slack_from_cust = self.model.binary_var()
            self.model.add_constraint(self.model.sum(v for v in assign_list_cust_from) + slack_from_cust == 1
                                      ,ctname='c_{0}'.format(cust))
            self.model.no_delivery.append(slack_from_cust)
            
            # Constraint to restrict one path to a customer
            assign_list_cust_to = [v.route_assigned for v in self.model.deliveries if v.end_location == cust]
            slack_to_cust = self.model.binary_var()
            self.model.add_constraint(self.model.sum(v for v in assign_list_cust_to) + slack_to_cust == 1
                                      ,ctname='c_{0}'.format(cust))
            self.model.no_delivery.append(slack_to_cust)
            
            
            # Constraint to assign single driver for a route
            # for cust_from in cust_list:
            #     if cust != cust_from:
            #         driver_route_list = [k for k in self.model.deliveries if k.cust_to == cust and k.cust_from == cust_from]
            #         self.model.add_constraint(self.model.sum(v for v in driver_route_list) == 1)          
                    
        return
    
    def start_wait_time_constraints(self):
        deliveries = [k for k in self.model.deliveries if k.end_location != 'DEPOT01']
        df_driver_deliveries = pd.DataFrame(deliveries)
        self.model.total_wait_time = []
        self.model.total_late_time = []
        for _,row in df_driver_deliveries.iterrows():
            driver = row.driver
            # cust_from = row.start_location
            cust_to = row.end_location
            SL_starttime = row.startlocation_starttime
            EL_starttime = row.endlocation_starttime
            travel_time = row.travel_time
            assigned_var = row.route_assigned
            wait_time = row.wait_time
            serve_time = row.serve_time
            late_time = row.late_time
            # delivered_load = row.delivered_load
            # M = 40000
            
            #Wait time and start time constraints for start and end location
            # self.model.add_constraint(SL_starttime + serve_time + travel_time - (1-assigned_var)*M + wait_time <= EL_starttime)
            
            self.model.add_indicator(assigned_var,
                                     SL_starttime + serve_time + travel_time + wait_time <= EL_starttime, 1)
            
            # Contraint to restrict start time of customer after the earliest possible start time
            earliest_start_time = [i.earliest_delivery_time for i in self.model.order if i.customer == cust_to][0]
            self.model.add_indicator(assigned_var, EL_starttime >= earliest_start_time, 1)
            # self.model.add_constraint(EL_starttime >= earliest_start_time)
            
            # Constraint to assign late time based on latest delivery time
            latest_start_time = [i.latest_delivery_time for i in self.model.order if i.customer == cust_to][0]
            self.model.add_indicator(assigned_var, EL_starttime - late_time <= latest_start_time, 1)
            # self.model.add_contraint(EL_starttime - late_time <= latest_start_time)
            
            # Constraint to restrict start times, wait_time and late_time to 0 if route is not assigned to driver
            self.model.add_indicator(assigned_var,SL_starttime == 0,0)
            self.model.add_indicator(assigned_var,wait_time == 0,0)
            self.model.add_indicator(assigned_var,late_time == 0,0)
            self.model.add_indicator(assigned_var,EL_starttime == 0,0)
            # self.model.add_indicator(assigned_var,delivered_load == 0,0)
            
            # Append wait time and late time to list
            self.model.total_wait_time.append(wait_time)
            self.model.total_late_time.append(late_time)

            
            # Constraint to assign route only if driver is assigned to driver truck
            driver_truck_assign_var = self.model.get_driver_truck[driver][2]
            self.model.add_constraint(assigned_var <= driver_truck_assign_var)
            
            
        return 
        
    def truck_capacity_constraints(self):
        for driver in self.model.driver_list:
            driver_deliveries = [k for k in self.model.deliveries if k.driver == driver]
            from_depot = [k for k in self.model.deliveries if k.driver == driver and k.start_location == 'DEPOT01']
            # to_depot = [k for k in self.model.deliveries if k.driver == driver and k.cust_to == 'DEPOT01']
            
            truck_capacity = self.model.get_driver_truck[driver][1]
            driver_vol = 0
            for delivery in driver_deliveries:
                del_volume = self.model.get_order_volume[delivery.end_location]
                driver_vol += self.model.sum(del_volume*delivery.route_assigned)

            for delivery_depot in from_depot:
                slack_load = self.model.continuous_var(lb=0)
                self.model.add_indicator(delivery_depot.route_assigned, delivery_depot.current_load == truck_capacity - slack_load, 1)
                self.model.add_indicator(delivery_depot.route_assigned, delivery_depot.current_load == 0,0)
                
            # Constraint to restrict inbound and outbound to a customer to a single driver]
            cust_list = [v for v in self.model.unique_cust_list]
            for cust in cust_list:
                inbound_cust = [k for k in driver_deliveries if k.end_location == cust]
                outbound_cust = [k for k in driver_deliveries if k.start_location == cust]
                sum_inbound = self.model.sum(k.route_assigned for k in inbound_cust)
                sum_outbound = self.model.sum(k.route_assigned for k in outbound_cust)
                self.model.add_constraint(sum_inbound == sum_outbound)
                
            # Constraint to start and end at depot for each driver
            # leave_depot_dr = [v for v in driver_deliveries if v.cust_from == 'DEPOT01']
            # arrive_depot_dr = [v for v in driver_deliveries if v.cust_to == 'DEPOT01']
            # sum_leaving = self.model.sum(v.route_assigned for v in leave_depot_dr)
            # sum_arrive = self.model.sum(v.route_assigned for v in arrive_depot_dr)
            # self.model.add_constraint(sum_leaving == sum_arrive)
            

        return
    
    def volume_remain_constraints(self):
        # condition for customer to customer
        cust_list = [k for k in self.model.unique_cust_list if k != 'DEPOT01']
        for cust in cust_list:
            cust_endlocation_list = [v for v in self.model.deliveries if v.end_location == cust]
            cust_startlocation_list = [v for v in self.model.deliveries if v.start_location == cust]
            # cust_todepot_list = [v for v in self.model.deliveries if v.cust_from == cust and v.cust_to == 'DEPOT01']
            for end in cust_endlocation_list:
                cust_to_volume = self.model.get_order_volume[end.end_location]
                for start in cust_startlocation_list:
                    # Load remaining constraint from i to j if route assigned
                    self.model.add_if_then(self.model.logical_and(end.route_assigned,start.route_assigned) == 1,start.current_load == end.current_load - cust_to_volume)
                    # Starttime constraint for particular location
                    self.model.add_if_then(self.model.logical_and(end.route_assigned,start.route_assigned) == 1, end.endlocation_starttime == start.startlocation_starttime)
                    # self.model.add_indicator(self.model.logical_and(end.route_assigned,start.route_assigned), 
                    #                          start.delivered_load == end.delivered_load + self.model.get_order_volume[end.cus_to],1)

        cust_list_depot = [v for v in self.model.deliveries if v.end_location == 'DEPOT01']
        for depot_delivery in cust_list_depot:
            self.model.add_indicator(depot_delivery.route_assigned, depot_delivery.current_load == 0,1)
                    
        return
    
    
    def first_delivery_constraints(self):
        # Constraint for delivered load for first depot
        # Constraint for start time, end time, wait time for depot
        # - when its first delivery
        # - when its not first delivery

        return
    
    def add_constraints(self):
        self.assignment_constraint()
        print("Added assignment constraints for customer")
        self.start_wait_time_constraints()
        print("Added wait time, late time and start time constraints")
        self.truck_capacity_constraints()
        print("Added capacity constraints from depot")
        self.volume_remain_constraints()
        print("Added constraints to update volume at each customer")
     
        
class SetupObjectives(ModelObjects):
    
    def __init__(self, data, model: Model):
        self.input_json = data
        self.model = model
        
    def setup_objectives(self):
        self.model.total_no_delivery_slack = self.model.sum(self.model.no_delivery)
        self.model.sum_wait_time = self.model.sum(self.model.total_wait_time)
        self.model.sum_late_time = self.model.sum(self.model.total_late_time)
        
        self.model.add_kpi(self.model.sum_wait_time, "Total wait time by drivers: ")
        self.model.add_kpi(self.model.sum_late_time, "Total late time by drivers: ")
        
        self.model.minimize_static_lex([self.model.total_no_delivery_slack,
                                        self.model.sum_wait_time,
                                        self.model.sum_late_time])
        
        return
    
class SolveModel(ModelObjects):
    
    def __init__(self,json_input,model:Model):
        self.model = model
        self.json_input = json_input
        
    def parameter_set_with_timelimit(self, cplex, limit):
        ps = cplex.create_parameter_set()
        # ps.add(cplex.parameters.dettimelimit, limit[0])
        ps.add(cplex.parameters.timelimit, limit[0])
        ps.add(cplex.parameters.preprocessing.aggregator, limit[1]) # [40,20,20]
        ps.add(cplex.parameters.mip.polishafter.solutions, limit[2]) # [1,1,1]
        # ps.add(cplex.parameters.benders.strategy, limit[3]) # [0,3,3]
        # ps.add(cplex.parameters.barrier.algorithm, limit[3])# [0,3,3]
        # ps.add(cplex.parameters.mip.strategy.heuristicfreq, limit[4]) #[1,1,1]
        ps.add(cplex.parameters.mip.tolerances.mipgap, 0.1)
        # ps.add(cplex.parameters.barrier.algorithm, limit[3])
        # ps.add(cplex.parameters.mip.tolerances.mipgap, 0.1)
        return ps
        
    def solve_model(self):
        
        # print(self.model.var_counter)
        # f = open('model_x2_cmd.saved','w')
        # f.write('\n'.join(self.model.log_stored))
        
        self.model.export_as_lp(basename='model_x2_cmd',path='./')

        if not self.model.has_multi_objective():
            raise "Expecting a multi-objective model!"
        # Get the CPLEX instance from the docplex model
        self.model.cplex_ins = self.model.get_cplex()
        # Set default streams, to see something

        with open('cplex.log','w') as cplexlogs:
        # cplexlog = "cplex.log"
            self.model.cplex_ins.set_log_stream(cplexlogs)
            self.model.cplex_ins.set_results_stream(cplexlogs)
            # self.model.cplex_ins.runseeds(cnt=10)
            # Print the details of each multi-objective round
            self.model.cplex_ins.parameters.multiobjective.display.set(2)
            # model.cplex_ins.parameters.preprocessing.aggregator = 0
            # ps = [self.parameter_set_with_timelimit(self.model.cplex_ins, l) for l in [(937390,40,1,0),(1890885,20,1,3),(1865849,20,1,3)]]
            ps = [self.parameter_set_with_timelimit(self.model.cplex_ins, l) for l in [(1200,40,1),(300,20,1),(300,20,1)]]
            solve = self.model.cplex_ins.solve(paramsets=ps)
            self.model.cplex_ins.solution.write('model.sol')
            solution = SolveSolution.from_file('model.sol', self.model)[0]

        return solution 
    
class CreateResults(ModelObjects):
    
    def __init__(self,json_input,model:Model,solution):
        self.model = model
        self.json_input = json_input
        self.solution = solution
        
    def create_results(self): 
        
        #self.solution = SolveSolution.from_file('model.sol', self.model)[0]
    
        if self.solution:
            self.model.report_kpis(solution=self.solution)
            self.create_route_assigned()
        
        def create_route_assigned(self):
            cols = ['driver','startlocation_starttime','start_location','end_location','endlocation_starttime','assigned',
                                                                           'travel_time','wait_time','late_time','serve_time','current_load']
            routes_list = []
            for delivery in self.model.deliveries:
                if self.solution.get_value(delivery.route_assigned) >= 1e-8:
                    routes_list.append((delivery.driver,self.solution.get_value(delivery.startlocation_starttime),delivery.start_location,delivery.end_location,
                                        self.solution.get_value(delivery.endlocation_starttime),self.solution.get_value(delivery.route_assigned),delivery.travel_time,
                                        self.solution.get_value(delivery.wait_time),self.solution.get_value(delivery.late_time),delivery.serve_time,
                                        self.solution.get_value(delivery.current_load)))
            routes_df = pd.DataFrame(routes_list,columns=cols)
            routes_df.to_csv('routes.csv')
            
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
        print("Done setup data")
        self.Setup_Constraints = SetupConstraints(self.data, self.model)
        self.Setup_Constraints.add_constraints()
        print("Done setup constraints")
        self.Setup_Objectives = SetupObjectives(self.data, self.model)
        self.Setup_Objectives.setup_objectives()
        print("Done setup objectives")
        self.Model_Solve = SolveModel(self.data, self.model)
        self.solution = self.Model_Solve.solve_model()
        print("Finished solving model")
        self.Create_Results = CreateResults(
            self.data, self.model, self.solution)
        self.Create_Results.create_results()
        
        
if __name__ == "__main__":


    mb = ModelBuild('Routing_input_new.xlsx')
    mb.create_model_run()
    # send_high_level_result()
    # send_detail_result()
        