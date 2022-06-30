import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns 
import geopandas as gpd
import pandas as pd 
import numpy as np
import requests
import shapely
import pickle
import progressbar
import geopy.distance
from shapely.geometry import Point, Polygon, LineString 
from shapely.ops import nearest_points
from shapely.ops import transform
from shapely import geometry, ops
from functools import partial
from itertools import combinations
import geopandas as gpd
import pyproj
import os
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.sparse import lil_matrix
import itertools
import warnings
warnings.simplefilter('ignore')
import random
from random import choice, randint
import timeit
import time
import threading
from multiprocessing import Pool
from joblib import Parallel, delayed

crs_name = 'epsg:4326'
def build_network_data(GeoData):    
    """
    Returns node-dict and edge-dict which is networkx compatible network from Vegvesenets routable data available at 
    https://kartkatalog.geonorge.no/metadata/statens-vegvesen/nvdb-ruteplan-nettverksdatasett/8d0f9066-34f9-4423-be12-8e8523089313 
    
    Created by Samuel Berntzen: https://github.com/samuelberntzen
    """

    # Get unique nodes
    nodes = {}

    print("Beginning tonodes...")
    for i in GeoData.drop_duplicates('tonode').index:

        # Get node ID for filtering duplicates
        _id          = GeoData.iloc[i]['tonode']

        # Only keep unique observations
        if _id not in nodes:
            # Get attributes if applicable, else get centroid coordinate in linestring (error usually indicate roundabout as one edge)
            try:
                x        = GeoData.iloc[i]['geometry'].boundary[-1].x
                y        = GeoData.iloc[i]['geometry'].boundary[-1].y
                roadclass = int(GeoData.iloc[i]['funcroadclass'])
                isBridge = int(GeoData.iloc[i]['isbridge'])
                isTunnel = int(GeoData.iloc[i]['istunnel'])
                geometry = Point(x,y)
            except Exception as e:
                x        = GeoData.iloc[i]['geometry'].centroid.x
                y        = GeoData.iloc[i]['geometry'].centroid.y
                roadclass = int(GeoData.iloc[i]['funcroadclass'])
                isBridge = int(GeoData.iloc[i]['isbridge'])
                isTunnel = int(GeoData.iloc[i]['istunnel'])
                geometry = Point(x,y)

            # Save and append
            content = {'x':x,'y':y,'osmid':_id, 'roadclass': roadclass, 'isBridge': isBridge, 'isTunnel': isTunnel, 'geometry':geometry}
            nodes[_id] = content
        else:
            pass

    # DO SIMILAR FOR FROMNODE:
    print("Beginning fromnodes...")
    for i in GeoData.drop_duplicates('fromnode').index:
            # Get node ID for filtering duplicates
        _id         = GeoData.iloc[i]['fromnode']

        # Only keep unique observations
        if _id not in nodes:
            # Get attributes if applicable, else get random coordinate in linestring (error usually indicate roundabout as one edge)
            try:
                x        = GeoData.iloc[i]['geometry'].boundary[-1].x
                y        = GeoData.iloc[i]['geometry'].boundary[-1].y
                roadclass = int(GeoData.iloc[i]['funcroadclass'])
                # isBridge = int(GeoData.iloc[i]['isbridge'])
                # isTunnel = int(GeoData.iloc[i]['istunnel'])
                geometry = Point(x,y)
            except Exception as e:
                x        = GeoData.iloc[i]['geometry'].centroid.x
                y        = GeoData.iloc[i]['geometry'].centroid.y
                roadclass = int(GeoData.iloc[i]['funcroadclass'])
                isBridge = int(GeoData.iloc[i]['isbridge'])
                isTunnel = int(GeoData.iloc[i]['istunnel'])
                geometry = Point(x,y)

            # Save and append
            content = {'x':x,'y':y,'osmid':_id, 'roadclass': roadclass, 'isBridge': isBridge, 'isTunnel': isTunnel, 'geometry':geometry}
            nodes[_id] = content
        else:
            pass


    # Get edges into networkx format
    edges = {}
    print("Beginning edges...")
    for i in GeoData.index:
        # Lets keep edges undirected for now (not one way)
        # Get edge ID for filtering duplicates:
        _id                 = GeoData.iloc[i]['linkid']

        # Get only data of edges not already retrieved
        if _id not in edges:
            ref                 = GeoData.iloc[i]['streetname'] 
            funcroadclass       = GeoData.iloc[i]['funcroadclass']
            roadclass           = GeoData.iloc[i]['roadClass']
            isFerry             = GeoData.iloc[i]['isferry']
            isBridge            = GeoData.iloc[i]['isbridge']
            isTunnel            = GeoData.iloc[i]['istunnel']
            speedlim            = GeoData.iloc[i]['speedfw']
            drivetime           = GeoData.iloc[i]['drivetime_fw']
            oneway              = False if GeoData.iloc[i]['oneway'] == "B" else False
            geometry            = GeoData.iloc[i]['geometry']
            u                   = GeoData.iloc[i]['fromnode']
            v                   = GeoData.iloc[i]['tonode']
            key                 = 0

            # linestring_trans = transform(project, GeoData.iloc[i]['geometry'])
            length = GeoData.iloc[i]['length'] - isFerry * GeoData.iloc[i]['length']
            length_weight = length.copy()

            # Estimate length based on speedlimit and drivetime
            # length_estimated = speedlim*drivetime*1000/60

            # Create dictionary of node data:
            content = {'id':_id, 'oneway':oneway, 'ref':ref, 'name':ref, 'funcroadclass':funcroadclass, 'roadclass':roadclass, 'isFerry':isFerry, 'isBridge':isBridge, 'isTunnel':isTunnel, 'speedlim':speedlim, 'drivetime':drivetime, 'length':length, 'length_weight':length_weight, 'geometry':geometry,'u':u, 'v':v, "key": key}

            edges[(u,v,0)] = content
        else:
            pass 

    # Set crs system
    crs = {'init': crs_name}

    # Create for nodes
    nodes_df = gpd.GeoDataFrame(nodes, crs = crs).T
    nodes_df = gpd.GeoDataFrame(
        nodes_df, geometry=nodes_df['geometry'])

    # Create for edges
    edges_df = gpd.GeoDataFrame(edges, crs = crs).T
    edges_df = gpd.GeoDataFrame(
        edges_df, geometry=edges_df['geometry'])


    return nodes_df, edges_df

def get_edge_attributes(cumlist):
    
    
    """
    Function that takes a list of network nodes and merge their attributes as the different attributes need to be treated differently.
    
    Input:
    cumlist: Cummulative list of nodes
    
    """

    cumlength = len(cumlist)
    path_attributes = dict()
    attrs_to_sum = {"length", "drivetime",}
    attrs_to_set = {'oneway', 'ref', 'name', 'funcroadclass', 'roadclass'}
    dummy_attrs = {'isFerry', 'isTunnel', 'isBridge'}
    nodes_to_remove = []
    
    for i in range(cumlength-1):
        u = cumlist[i]
        v = cumlist[i+1]
        edge_data = H.edges[u, v,0]

        for attr in edge_data:
            if attr in path_attributes:
            # if this key already exists in the dict, append it to the value list
                path_attributes[attr].append(edge_data[attr])
            else:
            # if this key doesn't already exist, set the value to a list containing the one value
                path_attributes[attr] = [edge_data[attr]]

    for attr in path_attributes:
        if attr == 'grade' or attr == 'length_weight':
            path_attributes[attr] = list(path_attributes[attr]) 
        elif attr in dummy_attrs:
            #If attribute is isFerry, isBridge og isTunnel, returns the value for first and last edge as these are candidate locations.
            if type(path_attributes[attr][0]) == list and type(path_attributes[attr][-1]) == list:
                fromnode = path_attributes[attr][0][0]
                tonode = path_attributes[attr][-1][-1]
                
            elif type(path_attributes[attr][0]) != list and type(path_attributes[attr][-1]) != list:
                fromnode = path_attributes[attr][0]
                tonode = path_attributes[attr][-1]
                
            elif type(path_attributes[attr][0]) == list and type(path_attributes[attr][-1]) != list:
                fromnode = path_attributes[attr][0][0]
                tonode = path_attributes[attr][-1]
                
            elif type(path_attributes[attr][0]) != list and type(path_attributes[attr][-1]) == list:
                fromnode = path_attributes[attr][0]
                tonode = path_attributes[attr][-1][-1]
            path_attributes[attr] = list((fromnode, tonode))
            #path_attributes[attr] = list((path_attributes[attr][0], path_attributes[attr][-1]))
        elif attr in attrs_to_sum:
            # if this attribute must be summed, sum it now
            path_attributes[attr] = sum(path_attributes[attr])

        elif attr == 'id':
            # if there's only 1 unique value in this attribute list, consolidate it to the single value (the zero-th):
            path_attributes[attr] = list((cumlist[0],cumlist[-1]))

        
        elif attr == 'geometry':
                path_attributes[attr] = ops.linemerge(path_attributes[attr])
                
        elif attr == 'oneway':
            path_attributes[attr] = 'False'
        # elif attr in numpy_attrs:
        #     path_attributes[attr] = path_attributes[attr]

        else:
        # otherwise, if there are multiple values, keep one of each
            try:
                path_attributes[attr] = list(set(path_attributes[attr]))
            except TypeError:
                try:
                    path_attributes[attr] = list(set(path_attributes[attr][0]))
                except:
                    path_attributes[attr] = path_attributes[attr]
            
    path_attributes['artificial'] = 1
    return path_attributes

def shorten_edges_v2(G, OD, cutoff):
    
    """
    Simplifies the edges in the network in order to make the runtimes of the GA/greedy substitution feassible. Depending on the threshold value, the function will remove all nodes/edges within a certain range, store and merge
    their attributes, before creating a new edge between the first and last node in the specified range. If threshol is for example 50000, the function will remove all nodes except the first and last node for every 50th km and 
    create a new edge between the first and last node. This i then repeated for each path between each OD pair.
    
    There are also certain statements to make sure that the same nodes are used if several paths intersect. This usually happens when multiple routes shares the same path. 
    
    Input:
    G: Networkx Multigraph
    OD: CSV file with coordinates of each OD location in Norway
    Threshold: Cutoff in meters at which edges at merged. 
    """

    
    #Create flat list of O-D pairs
    OD_pairs_flat = [item for sublist in OD_pairs for item in sublist]
    
    #Create empty MultiGraph
    B = nx.MultiGraph(crs='epsg:4326')
    
    #Initialize progressbar to monitor progress of function while running.
    maxval = len(OD_pairs)
    bar    = progressbar.ProgressBar(maxval=maxval, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    counter=0
    bar.start()

    UsedNodes = []
    for i in OD_pairs:
        fromOD         = i[0]
        toOD           = i[1]
        path           = nx.shortest_path(H, fromOD, toOD, 'length')
        pathlength     = len(path)
        lengthofpath   = nx.shortest_path_length(H, fromOD, toOD, 'length')
        cumlength      = 0
        cumlist        = []
        cutoff         = cutoff


        #for each in edge in path[i]
        for i in range(pathlength-1):
            fromnode = path[i]
            tonode = path[i+1]
            templength = H.edges[fromnode, tonode, 0]['length']

            cumlist.append(fromnode)
            cumlist.append(tonode)
            cumlength += templength

            #First check if tonode already exists in B. If it does, it means that a previously created path has used same route. 
            #In this case we want to use existing nodes on path rather than creating new ones.
            if tonode in UsedNodes:
                #If edge does not exist, then add it
                if B.has_edge(cumlist[0], cumlist[-1]) is False:
                    cumlist = list(dict.fromkeys(cumlist))
                    #Get attributes for nodes and edge to add
                    fnodeattr = H.nodes[cumlist[0]]
                    tnodeattr = H.nodes[cumlist[-1]]
                    edgeattr = get_edge_attributes(cumlist)

                    #Add nodes and edge
                    B.add_node(cumlist[0], **fnodeattr)
                    B.add_node(cumlist[-1], **tnodeattr)
                    B.add_edge(cumlist[0], cumlist[-1], **edgeattr)

                    #Added nodes are appended to all UsedNodes
                    UsedNodes.append(cumlist[0])
                    UsedNodes.append(cumlist[-1])
                    cumlist = []
                    cumlength = 0


                #Else, skip this part, but empty cummulative path and length
                else:
                    cumlist = []
                    cumlength = 0
                    pass
                
            elif tonode in AllCSnodes: #if next node is a CS node
                #If edge does not exist, then add it
                if B.has_edge(cumlist[0], cumlist[-1]) is False:
                    cumlist = list(dict.fromkeys(cumlist))
                    #Get attributes for nodes and edge to add
                    fnodeattr = H.nodes[cumlist[0]]
                    tnodeattr = H.nodes[cumlist[-1]]
                    edgeattr = get_edge_attributes(cumlist)

                    #Add nodes and edge
                    B.add_node(cumlist[0], **fnodeattr)
                    B.add_node(cumlist[-1], **tnodeattr)
                    B.add_edge(cumlist[0], cumlist[-1], **edgeattr)

                    #Added nodes are appended to all UsedNodes
                    UsedNodes.append(cumlist[0])
                    UsedNodes.append(cumlist[-1])
                    cumlist = []
                    cumlength = 0


                #Else, skip this part, but empty cummulative path and length
                else:
                    cumlist = []
                    cumlength = 0
                    pass
                

            #Next, check if tonode is an OD node. If it is we will create edge from start of cumlist to the OD node.
            elif tonode in OD_pairs_flat:
                #If edge does not exist, then add it
                if B.has_edge(cumlist[0], cumlist[-1]) is False:
                    cumlist = list(dict.fromkeys(cumlist))
                    #Get attributes for nodes and edge to add
                    fnodeattr = H.nodes[cumlist[0]]
                    tnodeattr = H.nodes[cumlist[-1]]
                    edgeattr = get_edge_attributes(cumlist)

                    #Add nodes and edge
                    B.add_node(cumlist[0], **fnodeattr)
                    B.add_node(cumlist[-1], **tnodeattr)
                    B.add_edge(cumlist[0], cumlist[-1], **edgeattr)

                    UsedNodes.append(cumlist[0])
                    UsedNodes.append(cumlist[-1])
                    cumlist = []
                    cumlength = 0
                #Else, skip this part, but empty cummulative path and length
                else:
                    cumlist = []
                    cumlength = 0
                    pass

            #Next, check if the cummulative length in path is higher than the set threshold. If it is we will create a new edge in B
            #Can consider adding an if inside here that check if there is a node within a certain radius, and skip if it is.
            #This could prevent that nodes on completely different paths are very close or that a node is set very close to an OD node. 
            #Would then find last element in cumlist in H and get lat,lon, then use ox nearest node to find nearest node in B and finally use geopy to find distance between them. 

            elif cumlength >= threshold:            


                if B.has_edge(cumlist[0], cumlist[-1]) is False:
                    #Remove duplicates from cumlist
                    cumlist = list(dict.fromkeys(cumlist))
                    distanceOD = 100000
                    #Before we add new edge, check if destination node is within threshold/2
                    lat = H.nodes[cumlist[-1]]['y']
                    lon = H.nodes[cumlist[-1]]['x']
                    latlon = (lat,lon)
                    latdestination = H.nodes[toOD]['y']
                    londestination = H.nodes[toOD]['x']

                    #Also check if there is an already created node nearby
                    try:
                        potentialnode = ox.get_nearest_node(B, latlon)
                        clatclon = (latdestination,londestination)
                        distanceOD = (geopy.distance.geodesic(latlon,clatclon).m)

                    except ValueError:
                        pass

                    # clatclon = (latdestination,londestination)
                    # distanceOD = (geopy.distance.geodesic(latlon,clatclon).m)
                    # distancepath = geopy.distance.geodesic(latlon, platplon).m

                    if distanceOD <= threshold/2:
                        pass


                    else:
                        #Get attributes for nodes and edge to add
                        fnodeattr = H.nodes[cumlist[0]]
                        tnodeattr = H.nodes[cumlist[-1]]
                        edgeattr = get_edge_attributes(cumlist)

                        #Add nodes and edge
                        B.add_node(cumlist[0], **fnodeattr)
                        B.add_node(cumlist[-1], **tnodeattr)
                        B.add_edge(cumlist[0], cumlist[-1], **edgeattr)

                        UsedNodes.append(cumlist[0])
                        UsedNodes.append(cumlist[-1])                
                        cumlist = []
                        cumlength = 0

                #Else, skip this part, but empty cummulative path and length
                else:
                    cumlist = []
                    cumlength = 0
                    pass

        bar.update(counter)
        counter+=1

    bar.finish()

    #Set OD=1 for all O-D nodes and CS=1 for all current CS nodes. 
    nx.set_node_attributes(B, 0, 'OD')
    nx.set_node_attributes(B, 0, 'CS')

    attrs = {}
    for i in OD_pairs_flat:
        O = i
        attrs[i] = {'OD':1}
    nx.set_node_attributes(B, attrs)
    
    attrs = {}
    for i in AllCSnodes:
        O = i
        attrs[i] = {'CS':1}
    nx.set_node_attributes(B, attrs)
    
    
    print("Complete!")
    return B



def calculate_eff_lengths(edge):
    
    """
    Calculates the effective length of an edge by using length weights and road gradient
    """

    startnode = edge[0]
    endnode = edge[1]
    const = 0.372
    grade_intervals = [-0.09, -0.07, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11]
    increased_consumption = [-0.332, -0.217, -0.148, -0.121, -0.073, 0.085, 0.152, 0.203, 0.306, 0.358, 0.552]
    
    iterator = len(B.edges[edge]['length_weight'])
    
    #If the direction of the edge is the default fromnode -> tonode, we set a multiplier to 1, meaning that the edge gradients will be default:
    if B.nodes[startnode]['elevation'] < B.nodes[endnode]['elevation']: 
        direction = 1
    #If direction is reversed, meaning that the vehicle is returning to the origin node, we reverse the gradients
    else:
        direction = -1
    eff_length_trueway = 0
        
    for i in range(iterator):
        edge_grad = B.edges[edge]['grade'][i]*(direction)
        length_weight = B.edges[edge]['length_weight'][i]

        if edge_grad < grade_intervals[0]: #less than -9%
            diff = const + increased_consumption[0]
            eff_length_trueway += length_weight*(diff/const)

        elif grade_intervals[0] < edge_grad <= grade_intervals[1]: #-9% to -7%
            diff = const + increased_consumption[1]
            eff_length_trueway += length_weight*(diff/const)

        elif grade_intervals[1] < edge_grad <= grade_intervals[2]: #-7% to -5%
            diff = const + increased_consumption[2]
            eff_length_trueway += length_weight*(diff/const)

        elif grade_intervals[2] < edge_grad <= grade_intervals[3]: #-5% to -3%
            diff = const + increased_consumption[3]
            eff_length_trueway += length_weight*(diff/const)

        elif grade_intervals[3] <= edge_grad <= grade_intervals[4]: #-3% to -1%
            diff = const + increased_consumption[4]
            eff_length_trueway += length_weight*(diff/const)

        elif grade_intervals[5] <= edge_grad <= grade_intervals[6]: #1% to #3%
            diff = const + increased_consumption[5]
            eff_length_trueway += length_weight*(diff/const)

        elif grade_intervals[6] < edge_grad <= grade_intervals[7]: #3% to 5%
            diff = const + increased_consumption[6]
            eff_length_trueway += length_weight*(diff/const)

        elif grade_intervals[7] < edge_grad <= grade_intervals[8]: #5% to 7%
            diff = const + increased_consumption[7]
            eff_length_trueway += length_weight*(diff/const)

        elif grade_intervals[8] < edge_grad <= grade_intervals[9]: #7% to 9%
            diff = const + increased_consumption[8]
            eff_length_trueway += length_weight*(diff/const)

        elif grade_intervals[9] < edge_grad <= grade_intervals[10]: #9% to 11%
            diff = const + increased_consumption[9]
            eff_length_trueway += length_weight*(diff/const)

        elif edge_grad > grade_intervals[10]: #more then 11%
            diff = const + increased_consumption[0]
            eff_length_trueway += length_weight*(diff/const)


        else: #If grade is between -1% and 1%
            eff_length_trueway += length_weight
            diff = 0
     
    return eff_length_trueway


def GravityModel(master_df):
    flow=[]
    master_df['flow']=0
    for i in master_df.index:
        GC=(0.75*(master_df['length'][i])+0.5*master_df['drivetime'][i])
        flow.append(master_df['weight'][i]/GC)
    
    tot_flow=sum(flow)
    for i in master_df.index:
        master_df['flow'][i]=flow[i]/tot_flow
GravityModel(df)



def Objectivefunction(FacilitiesList, paths, flows, lengths_dict, G_dict, Totalrange):
    startime = time.time()
    CoveredObjectiveValue = 0
    facilities = tuple(FacilitiesList)
    
    for j in range(len(paths)):
        path = tuple(paths[j])
        NodeInFacilityListExistInPath = False
        if (lengths_dict[path[0],path[-1]]+lengths_dict[path[-1],path[0]]) <=Totalrange/2: # If the round trip length is shorter than half the EV range, add the flow immediately 
            CoveredObjectiveValue += flows[j]
        
        else:
            for node in path:
                if node in facilities:
                    NodeInFacilityListExistInPath = True
                    break
                    
                    
        if NodeInFacilityListExistInPath == True: #Only evaluate if at least one facility is on the path
            
            # Take first from Origin to Destination:
            if path[0] in facilities:
                rem_range = Totalrange
            else:
                rem_range = Totalrange/2
                
            indicator_canRefuel=0
                
            #Start looping through the path going first from origin to the destination:
            revpath = path[::-1][1:]
            fullpath = path+revpath                    
            
            for i in range(len(fullpath)+1):                
                a = fullpath[i]
                b = fullpath[i+1]                                                                                     
                rem_range -= G_dict[(a,b)]

                if rem_range <0:
                    break
                elif b == path[-1] & path[-1] in facilities:
                    indicator_canRefuel+=1
                    break
                elif b == path[0]:
                    indicator_canRefuel+=1
                    break
                elif b in facilities:
                    rem_range = Totalrange
            
            if indicator_canRefuel==1:
                
                
                # Now consider from Destination to Origin:
                dest_orig_path = path[::-1]
                fullpath = dest_orig_path + path[1:]

                if fullpath[0] in facilities:
                    rem_range = Totalrange
                else:
                    rem_range = Totalrange/2

                for i in range(len(fullpath)+1):            
                    a = fullpath[i]
                    b = fullpath[i+1]                                                                                     
                    rem_range -= G_dict[(a,b)]

                    if rem_range <0:
                        break
                    elif b == dest_orig_path[-1] & dest_orig_path[-1] in facilities:
                        indicator_canRefuel+=1
                        break
                    elif b == dest_orig_path[0]:
                        indicator_canRefuel+=1
                        break
                    elif b in facilities:
                        rem_range = Totalrange               

                if indicator_canRefuel==2:
                    CoveredObjectiveValue += flows[j]
    
                
    # endtime = time.time()
    # finaltime = endtime-startime
    # print('B2: '+str(finaltime))
    return CoveredObjectiveValue


def greedysub(p, k, Totalrange, sub_iterations, flows, paths, lengths_dict, G_dict):
    
    starttime = time.time()
    selectedFacilities = []
    for z in range(p):
        print(z)
        CurrentMaxObjectiveValue = 0
        CurrentMaxFacility = 0
        for facility in k:
            FacilitiesList = selectedFacilities.copy()
            if facility not in selectedFacilities:
                FacilitiesList.append(facility)
                TempObjectiveValue = Objectivefunction(FacilitiesList, paths, flows, lengths_dict, G_dict, Totalrange)
                if (TempObjectiveValue > CurrentMaxObjectiveValue):
                    CurrentMaxObjectiveValue = TempObjectiveValue
                    CurrentMaxFacility = facility
        selectedFacilities.append(CurrentMaxFacility)
        
        for i in range(sub_iterations):
            SubFacilityIndex = list(np.setdiff1d(k,selectedFacilities))
            SubMaxObjectiveValue = 0
            SubMaxFacility = 0
            IndexBeingSubstituted = 0
            for index in range(len(selectedFacilities)):
                if selectedFacilities[index] != CurrentMaxFacility:
                    TempFacilities = selectedFacilities.copy()
                    TempFacilities.remove(selectedFacilities[index])
                    for Subfacility in SubFacilityIndex:
                        SubArray = TempFacilities.copy()
                        SubArray.append(Subfacility)
                        CurrentObjectiveValue = Objectivefunction(SubArray, paths, flows, lengths_dict, G_dict, Totalrange)
                        if CurrentObjectiveValue > SubMaxObjectiveValue:
                            SubMaxObjectiveValue = CurrentObjectiveValue
                            SubMaxFacility = Subfacility
                            IndexBeingSubstituted = index                        
            
            if SubMaxObjectiveValue > CurrentMaxObjectiveValue: 
                selectedFacilities[IndexBeingSubstituted] = SubMaxFacility
                CurrentMaxObjectiveValue = SubMaxObjectiveValue
                CurrentMaxFacility = SubMaxFacility
            else:
                break
            
    endtime = time.time()
    print("runtime",round(endtime-starttime,1),"seconds")
    return selectedFacilities, CurrentMaxObjectiveValue


def random_combination(k, p, pop_limit):
    pool = [i for i in k]
    population = []
    rest_filled=False
    while len(population) < pop_limit:
        if len(pool)>=p:
            chromosome_i = random.sample(pool, p)
            population.append(chromosome_i)
            for ind in chromosome_i:
                pool.remove(ind)            
        elif rest_filled==False:            
            fill_rest=random.sample(k,p-len(pool))
            population.append(pool+fill_rest)
            rest_filled=True
        else:
            chromosome_i = sorted(random.sample(k, p))            
            if chromosome_i not in population:
                population.append(chromosome_i)
    return population


def Union(father, mother):
    """
    Inputs two lists and return the union of these lists
    
    """
    unionlist= list(set(father) | set(mother))
    return unionlist


def crossover(f, union, flows, paths, lengths_dict, G_dict):
    """
    Performs the greedy deletion crossover algorithm which is a part of the genetic algorithm
    
    input:
    f: iterator, f goes from 0 to length of union
    union: unioned chromosones from father and mother
    """
    
    A = union[f]
    totalgenes = len(A)
    for i in range(totalgenes, p, -1):
        CurrentMaxObjectiveValue = 0
        CurrentMaxIndex = 0
        for j in range(len(A)):
            TempFacilities = A.copy()
            TempFacilities.pop(j)                    
            TempObjectiveValue = Objectivefunction(TempFacilities, paths, flows, lengths_dict, G_dict, Totalrange)
            if TempObjectiveValue > CurrentMaxObjectiveValue:
                CurrentMaxObjectiveValue = TempObjectiveValue
                CurrentMaxIndex = j
        A.pop(CurrentMaxIndex)
    children1 = A
    return children1



def genetic(k, p, pop_limit, mutation_freq, nr_gen, Totalrange, flows, paths, lengths_dict, G_dict):
    
    """
    Genetic Algorithm. Algorithm that is modeled after the process of evolution. Starts with multiple sets of chromosones (random solutions) forming a population. Each chromosone consists of p genes (facilities). 
    For each generation, better solutions are produced until the maximum number of generations are reached of if the algorithm converges.
    
    Input:
    k: List of candidate locations
    p: Number of facilities to be located
    pop_limit: Starting population/population limit
    mutation_freq: Chance of gene mutation
    nr_gen: Number of generations
    Totalrange: Full range of the electric vehicles when fully recharged
    
    
    USED FOR CALIBRATION PURPOSES
    """
    #Initialize the maximum objective value and a list of all maximum objective values for each generation. Also initiate timer so we can measure the time spent
    MaxObjectiveValue = 0
    AllObjectiveValues = []
    start = time.time()
    
    #Initialize an empty list where the facilities that captures the most flow are stored
    best_pop = []
    
    #Initialize population
    population = random_combination(k, p, pop_limit)
    
    #Selecting the parents
    for x in range(nr_gen):
        mother = random.sample(population, round(pop_limit/2))
        for i in mother: 
            population.remove(i)
            
        father = population.copy()   

        #Union chromosones of mother and father
        iterations = len(father)
        union = [] 
        for i in range(iterations):
            temp = Union(father[i], mother[i])
            union.append(temp)
        #Greedy deletion crossover algorithm for each cromosone. Calls the crossover function and utilizes all available cores. This causes the algorithm to run about 200% faster
        start1=time.time()
        if __name__ == '__main__':
            children = Parallel(n_jobs=-1)(delayed(crossover)(f, union, flows, paths, lengths_dict, G_dict) for f in range(len(union)))
    
        end1=time.time()
        #New member selection. Select the two best out of children, mother and father, each with a length of pop_limit/2. Population length should then be = pop_limit
        population = []
        objectivevalues = []
        for i in range(round(pop_limit/2)):
            MotherValue = Objectivefunction(mother[i], paths, flows, lengths_dict, G_dict, Totalrange)
            FatherValue = Objectivefunction(father[i], paths, flows, lengths_dict, G_dict, Totalrange)
            ChildrenValue = Objectivefunction(children[i], paths, flows, lengths_dict, G_dict, Totalrange)
            minValue = min(MotherValue, FatherValue, ChildrenValue)
            maxValue = max(MotherValue, FatherValue, ChildrenValue)
            if MotherValue == minValue:
                population.append(list(father[i])) 
                population.append(list(children[i]))
                objectivevalues.append(FatherValue)
                objectivevalues.append(ChildrenValue)
            elif FatherValue == minValue:
                population.append(list(mother[i]))
                population.append(list(children[i]))
                objectivevalues.append(MotherValue)
                objectivevalues.append(ChildrenValue)            
            elif ChildrenValue == minValue:
                population.append(list(mother[i]))
                population.append(list(father[i]))
                objectivevalues.append(FatherValue)
                objectivevalues.append(MotherValue)
            
            #Picks the highest scoring chromosone and stores it as the best solution for current generation
            if MotherValue == maxValue:
                if MotherValue > MaxObjectiveValue:
                    MaxObjectiveValue = MotherValue
                    best_pop = mother[i]

            elif FatherValue == maxValue:
                if FatherValue > MaxObjectiveValue:
                    MaxObjectiveValue = FatherValue
                    best_pop = father[i]

            elif ChildrenValue == maxValue:               
                if ChildrenValue > MaxObjectiveValue:
                    MaxObjectiveValue = ChildrenValue
                    best_pop = children[i]                
        #Mutation, do not mutate last iteration
        if x < (nr_gen-1):
            for i in range(len(population)): #For each Chromosome
                if random.random() < mutation_freq: #If a chrosome is chosen one random gene in it will mutate 
                    j = randint(0, p-1)
                    population[i][j] = random.sample(k,1)[0]


        AllObjectiveValues.append(MaxObjectiveValue)

        """If best_pop flow has not improved the last three generations it is safe to assume that it has converged and the loop can break"""
        if len(AllObjectiveValues) > 4:
            if AllObjectiveValues[x] == AllObjectiveValues[(x-1)] == AllObjectiveValues[x-2] == AllObjectiveValues[x-3]:
                print("Convergence reached at gen:",(x+1))
                print(round(MaxObjectiveValue,2),"% of flow captured")
                end = time.time()
                print("runtime",round(end-start,1),"seconds")
                
                return MaxObjectiveValue, (x+1), (round(end-start,1))
            
    """If no convergence has occured, the algorithm will finish once it has completed all generations and return the best solution it could find"""
    # print("Maximum number of generations reached")
    # print("Flow captured:", MaxObjectiveValue)
    end = time.time()
    print("runtime",round(end-start,1),"seconds")
    final = end-start
    conv_gen = AllObjectiveValues.index(MaxObjectiveValue)+1
    #does not return best population. used for calibration
    return MaxObjectiveValue, final, conv_gen

def genetic_pop(k, p, pop_limit, mutation_freq, nr_gen, Totalrange, flows, paths, lengths_dict, G_dict, solution_co, indicat):
    """
    Genetic Algorithm. Algorithm that is modeled after the process of evolution. Starts with multiple sets of chromosones (random solutions) forming a population. Each chromosone consists of p genes (facilities). 
    For each generation, better solutions are produced until the maximum number of generations are reached of if the algorithm converges.
    
    Input:
    k: List of candidate locations
    p: Number of facilities to be located
    pop_limit: Starting population/population limit
    mutation_freq: Chance of gene mutation
    nr_gen: Number of generations
    Totalrange: Full range of the electric vehicles when fully recharged
    
    
    USED FOR FINDING BEST SOLUTIONS
    """
#Initialize the maximum objective value and a list of all maximum objective values for each generation. Also initiate timer so we can measure the time spent
    MaxObjectiveValue = 0
    AllObjectiveValues = []
    count_escape = 0
    start = time.time()
    
    #Initialize an empty list where the facilities that captures the most flow are stored
    best_pop = []
    
    #Initialize population
    if p*pop_limit<len(k):        
        init_pop_size=math.ceil(math.ceil((1583)/p)/2)*2
        population = random_combination(k, p, init_pop_size)
    else:
        population = random_combination(k, p, pop_limit)
        
    #inlcude solution_co in initial population:
    if indicat==1:
        population.append(solution_co)                
    
    #Selecting the parents
    for x in range(nr_gen):
        start_i=time.time()
        mother = random.sample(population, math.ceil(len(population)/2))
        for i in mother: 
            population.remove(i)        
        father = population.copy()        
        if len(mother)-len(father)==1:
            father.append(random_combination(k, p, 1)[0])
        
        #Union chromosones of mother and father
        union = [] 
        for i in range(len(father)):
            temp = Union(father[i], mother[i])
            union.append(temp)
        #Greedy deletion crossover algorithm for each cromosone. Calls the crossover function and utilizes all available cores. This causes the algorithm to run about 200% faster        
        if __name__ == '__main__':
            children = Parallel(n_jobs=-1)(delayed(crossover)(f, union, flows, paths, lengths_dict, G_dict) for f in range(len(union)))
    
        
        #New member selection. Select the two best out of children, mother and father, each with a length of pop_limit/2. Population length should then be = pop_limit
        population = []
        objectivevalues = []
        for i in range(len(father)):
            MotherValue = Objectivefunction(mother[i], paths, flows, lengths_dict, G_dict, Totalrange)
            FatherValue = Objectivefunction(father[i], paths, flows, lengths_dict, G_dict, Totalrange)
            ChildrenValue = Objectivefunction(children[i], paths, flows, lengths_dict, G_dict, Totalrange)
            minValue = min(MotherValue, FatherValue, ChildrenValue)
            maxValue = max(MotherValue, FatherValue, ChildrenValue)
            if MotherValue == minValue:
                population.append(list(father[i])) 
                population.append(list(children[i]))
                objectivevalues.append(FatherValue)
                objectivevalues.append(ChildrenValue)
            elif FatherValue == minValue:
                population.append(list(mother[i]))
                population.append(list(children[i]))
                objectivevalues.append(MotherValue)
                objectivevalues.append(ChildrenValue)            
            elif ChildrenValue == minValue:
                population.append(list(mother[i]))
                population.append(list(father[i]))
                objectivevalues.append(FatherValue)
                objectivevalues.append(MotherValue)
            
            #Picks the highest scoring chromosone and stores it as the best solution for current generation
            if MotherValue == maxValue:
                if MotherValue > MaxObjectiveValue:
                    MaxObjectiveValue = MotherValue
                    best_pop = mother[i]

            elif FatherValue == maxValue:
                if FatherValue > MaxObjectiveValue:
                    MaxObjectiveValue = FatherValue
                    best_pop = father[i]

            elif ChildrenValue == maxValue:               
                if ChildrenValue > MaxObjectiveValue:
                    MaxObjectiveValue = ChildrenValue
                    best_pop = children[i]
        
        # Keep the best chromosomes to uphold the pop limit
        if len(population)>(pop_limit+1):                                  
            ob_values=[]           
            for i in range(len(population)):
                ob_values.append(Objectivefunction(population[i], paths, flows, lengths_dict, G_dict, Totalrange))
            fram=pd.DataFrame(ob_values)
            limit=pop_limit+(round((init_pop_size-pop_limit)/2))
            if x==0:
                keep=fram.sort_values(0, ascending=False)[0:limit]
            if x==1:
                keep=fram.sort_values(0, ascending=False)[0:pop_limit]
            indexes=list(keep.index)
            population=[population[i] for i in indexes]
            
        #Mutation, do not mutate last iteration
        if x < (nr_gen-1):            
            for i in range(len(population)): #For each Chromosome
                if random.random() < mutation_freq: #If a chrosome is chosen one random gene in it will mutate 
                    j = randint(0, p-1)
                    sub_in=random.sample(k,1)[0]
                    if sub_in not in population[i]:
                        population[i][j] = random.sample(k,1)[0]
                    else:
                        population[i][j] = random.sample(k,1)[0]
            
            # for i in range(len(population)): #For each Chromosome
            #     for j in range(p):
            #         if random.random() < mutation_freq: #If a gene is chosen one random gene in it will mutate                         
            #             sub_in=random.sample(k,1)[0]
            #             if sub_in not in population[i]:
            #                 population[i][j] = random.sample(k,1)[0]
            #             else:
            #                 population[i][j] = random.sample(k,1)[0]


        end_i=time.time()
        print("Generation:", (x+1), 'Flow:', MaxObjectiveValue, round(end_i-start_i,1))
        print(best_pop)
        AllObjectiveValues.append(MaxObjectiveValue)

        """If best_pop flow has not improved the last three generations it is safe to assume that it has converged and the loop can break"""        
        if len(AllObjectiveValues) > 5:
            if AllObjectiveValues[x] == AllObjectiveValues[(x-1)] == AllObjectiveValues[x-2]== AllObjectiveValues[x-3]:                                          
        
                for i in range(len(AllObjectiveValues)):
                    if AllObjectiveValues[i]==MaxObjectiveValue:
                        first_max_found=i+1
                        break
                end = time.time()                
                print(round(end-start,1), 'sek')
                # return MaxObjectiveValue, round(end-start,1), first_max_found
        
        
                return MaxObjectiveValue, best_pop, (round(end-start,1))
                
                
                
        # print("Generation:", (x+1), "Flow:", MaxObjectiveValue, count_escape)
            
    """If no convergence has occured, the algorithm will finish once it has completed all generations and return the best solution it could find"""
    # print("Maximum number of generations reached")
    # print("Flow captured:", MaxObjectiveValue)
    
    end = time.time()
    print(round(end-start,1), 'sek')
    for i in range(len(AllObjectiveValues)):
        if AllObjectiveValues[i]==MaxObjectiveValue:
            first_max_found=i+1
            break
    # return MaxObjectiveValue, round(end-start,1), first_max_found
    ######## mutation calibration: #######
   
    return MaxObjectiveValue, best_pop, (round(end-start,1))