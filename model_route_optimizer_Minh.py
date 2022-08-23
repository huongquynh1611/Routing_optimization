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
        self.model.TCapacity = namedtuple(
            'TCapacity', ["TruckID", 'Capacity', 'Units', 'DepotID'])
        self.model.TDepot = namedtuple(
            'TDepot', ['ID', 'Longitude', 'Latitude', 'TimeService'])
        self.model.TEmployee = namedtuple(
            'TEmployee', ['EmployeeID', 'Type', 'StartTime', 'EndTime', 'DepotID'])
        self.model.TOrder = namedtuple(
            'TOrder', ['OrderID', 'StoreID', 'Case', 'StartTime', 'EndTime'])
        self.model.TStore = namedtuple(
            'TStore', ['ID', 'Longitude', 'Latitude', 'Time_Service'])
        self.model.TConvert = namedtuple('TConvert', ['From', 'To', 'Rate'])
        # tuple for variables
        self.model.TROUTE = namedtuple(
            'TROUTE', ['from_store', 'to_store', 'distance', 'vehicle_id', 'depot_id'])
        self.model.TRoute = namedtuple('TRoute', [
                                       'var', 'name', 'from_store', 'to_store', 'distance', 'day', 'vehicle_id', 'depot_id'])
        self.model.TTime = namedtuple('TTime', [
                                      'var', 'name', 'store', 'order_id', 'start_time', 'end_time', 'employee_id', 'type', 'case', 'depot_id'])
        self.model.TVar = namedtuple('TVar', [
                                     'order_id', 'store_id', "receipt_start", 'receipt_end', 'route', 'time', 'case'])
        return

    def read_input_data(self):
        file = open(self.json_input, 'r')
        contents = file.read()
        file.close()
        self.input_json = json.loads(contents)

    def load_input_data(self):
        
        self.model.pallet_to_cases = 150
        
        self.model.df_capacity = json_normalize(self.input_json['Capacity'])
        self.model.df_employee = json_normalize(self.input_json['Employee'])
        self.model.df_convert = json_normalize(self.input_json['Convert'])
        self.model.df_order = json_normalize(self.input_json['Order'])
        self.model.df_store = json_normalize(self.input_json['Store'])
        self.model.df_depot = json_normalize(self.input_json['Depot'])
        self.model.df_store = pd.concat(
            [self.model.df_store, self.model.df_depot])

        # convert to datetime
        self.model.df_order['Receipt_Time_Start'] = pd.to_datetime(
            self.model.df_order['Receipt_Time_Start'], format='%Y-%d-%m %H:%M:%S')
        self.model.df_order['Receipt_Time_End'] = pd.to_datetime(
            self.model.df_order['Receipt_Time_End'], format='%Y-%d-%m %H:%M:%S')
        self.model.df_order['Case'] = pd.to_numeric(self.model.df_order['Case'])
        self.model.day_list = list(
            set(self.model.df_order["Receipt_Time_Start"].dt.strftime('%Y-%m-%d')))

        self.model.df_employee['StartTime'] = pd.to_datetime(
            self.model.df_employee['StartTime'], format='%Y-%d-%m %H:%M:%S')
        self.model.df_employee['EndTime'] = pd.to_datetime(
            self.model.df_employee['EndTime'], format='%Y-%d-%m %H:%M:%S')
        self.model.df_order = self.model.df_order.sort_values(
            ['Receipt_Time_Start', 'Receipt_Time_End'])
        self.model.anchor_date = pd.to_datetime(
            self.model.df_employee['StartTime']).min()
        # convert to num
        self.list_day = list(
            set(self.model.df_order.Receipt_Time_Start.dt.strftime('%Y-%m-%d')))

        def convert_time_to_num(df, col):
            df.loc[:, col] = list(map(self.date2num, df[col]))
        convert_time_to_num(self.model.df_order, 'Receipt_Time_Start')
        convert_time_to_num(self.model.df_order, 'Receipt_Time_End')
        convert_time_to_num(self.model.df_employee, 'StartTime')
        convert_time_to_num(self.model.df_employee, 'EndTime')

    def load_model_object(self):
        CAPACITY = [self.model.TCapacity(
            *row) for _, row in self.model.df_capacity.iterrows()]
        EMPLOYEE = [self.model.TEmployee(
            *row) for _, row in self.model.df_employee.iterrows()]
        CONVERT = [self.model.TConvert(*row)
                   for _, row in self.model.df_convert.iterrows()]
        ORDER = [self.model.TOrder(*row)
                 for _, row in self.model.df_order.iterrows()]
        STORE = [self.model.TStore(*row)
                 for _, row in self.model.df_store.iterrows()]
        DEPOT = [self.model.TDepot(*row)
                 for _, row in self.model.df_depot.iterrows()]

        self.model.capacity = CAPACITY
        self.model.employee = EMPLOYEE
        self.model.convert = CONVERT
        self.model.order = ORDER
        self.model.store = STORE
        self.model.depot = DEPOT

    def date2num(self, dt):
        return int((dt - self.model.anchor_date).total_seconds() / 3600)

    def convert_to_datetime(self, hour):
        beginning_date = self.model.anchor_date
        i = math.floor(hour / 24)
        j = int(hour - 24 * i)
        date = beginning_date + timedelta(days=i)
        date_time = date + timedelta(hours=j)
        return date_time

    def get_distance(self, lat1, long1, lat2, long2):
        return (abs(math.sqrt((pow(lat1 - lat2, 2)) + (pow(long1-long2, 2)))))


class SetupData(ModelObjects):
    def __init__(self, data, model: Model):
        self.json_input = data
        self.model = model
        self.model.unload_rate = 2000  # 2000 cases/hour

    def create_model_objects(self):
        self.read_input_data()
        self.load_input_data()
        self.create_tuples()
        self.load_model_object()
        self.create_route_var()

    def is_overlap(self, a, b, r=0):
        """
        a: tuple of range, ex (1,5)
        b: tuple of range, ex (4,10)
        r: overlap size, ex 1
        return True
        """
        return a[0] < b[1]+r and b[0] < a[1]+r

    def get_orders_by_availability(self, start_time, end_time):
        a = (start_time, end_time)
        return [order 
                for order in self.model.order 
                if self.is_overlap(a, (order.StartTime, order.EndTime), order.Case / self.model.unload_rate)]

    def create_route_var(self):

        self.ROUTE = []
        for avail in self.model.employee:
            lst_orders = self.get_orders_by_availability(avail.StartTime, avail.EndTime)
            for order in lst_orders:
                ""
                

    def create_time_var(self):
        return


class SetupConstraints(ModelObjects):

    def __init__(self, data, model: Model):
        self.input_json = data
        self.model = model

    def route_constraint(self):

        return

    def add_constraints(self):
        self.route_constraint()
        print('Finished constraints')


class SetupObjectives(ModelObjects):
    def __init__(self, model: Model):
        self.model = model


class ModelSolver(ModelObjects):
    def __init__(self, model):
        self.model = model

    def solve_model(self):
        solve = self.model.solve(log_output=True)
        print("Model solve complete")
        return solve


class CreateResults(ModelObjects):
    def __init__(self, data, model, solution):
        self.input_json = data
        self.model = model
        self.solution = solution


class ModelBuild:

    def __init__(self, data):
        self.model = Model()
        self.model.start_time = time.time()
        self.data = data

    def create_model_run(self):
        self.SetupData = SetupData(self.data, self.model)
        self.SetupData.create_model_objects()
        self.Setup_constraints = SetupConstraints(self.data, self.model)
        self.Setup_constraints.add_constraints()
        # self.Setup_objectives = SetupObjectives(self.model)
        # self.ModelSolve = ModelSolver(self.model)
        # self.solutoin = self.ModelSolve.solve_model()
        # self.CreateResults = CreateResults(self.data,self.model,self.solutoin)


if __name__ == '__main__':
    start_time = time.time()
    mb = ModelBuild('data.json')
    mb.create_model_run()
    print("--- %s seconds ---" % (time.time() - start_time))
