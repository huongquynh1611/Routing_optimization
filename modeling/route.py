import cplex
from cplex.exceptions import CplexError
import sys
import pandas as pd
import numpy as np



latlng = ['latitude', 'longitude']
position = pd.read_csv('D:\\Quynh\\Routing_optimization\\vehicle-routing-problem-master\\data\\position.csv', index_col="City")
flighttime = pd.read_csv('D:\\Quynh\\Routing_optimization\\vehicle-routing-problem-master\\data\\flight_time.csv', index_col="City")
distance = pd.read_csv('D:\\Quynh\\Routing_optimization\\vehicle-routing-problem-master\\data\\distance.csv', index_col="City")
