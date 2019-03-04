# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:14:00 2019

@author: Chad
"""

import pandas as pd
import networkx as nx
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

#Read in data
data = pd.read_csv('terroristdata.csv')

#Set the index to start at 1.  This will help with the subsetted adjancency matrices
data.index = np.arange(1, len(data)+1)

#data info
print(data.shape)

'''
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
SUBSETTING
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
'''

'''
Function to remove characters from the column names
leaving only the numbers. This is required for the networkx
adjacency matrix graph.
'''
def onlyNum(df):
   names = df.columns.tolist()
   empty = []
   exp = re.compile('\D+')
   for i in names:
      new = exp.sub('', i)
      new = int(new)
      empty.append(new)
   df.columns = empty
   return df


#subset ORGANIZATION (all rows and only the ORGAN columns)
organ = data.iloc[:,1:33]
organ = onlyNum(organ)

#subset SCHOOL (all rows and only the SCHOOL columns)
school = data.iloc[:,33:58]
school = onlyNum(school)

#subset CLASS (all rows and only the CLASS columns. This is an adjancency matrix)
classmate = data.iloc[:,58:137]
classmate = onlyNum(classmate)

#subset COMMUN (all rows and only the COMMUN columns. This is an adjancency matrix)
commun = data.iloc[:,137:216]
commun = onlyNum(commun)

#subset KIN (all rows and only the KIN columns. This is an adjancency matrix)
kin = data.iloc[:,216:295]
kin = onlyNum(kin)

#subset TRAIN (all rows and only the TRAIN columns)
train = data.iloc[:,295:311]
train = onlyNum(train)

#subset EMPLOY (all rows and only the EMPLOY columns)
employ = data.iloc[:,311:321]
employ = onlyNum(employ)

#subset OPERAT (all rows and only the OPERAT columns)
operat = data.iloc[:,321:335]
operat = onlyNum(operat)

#subset FRIEND (all rows and only the FRIEND columns. This is an adjancency matrix)
friend = data.iloc[:,335:414]
friend = onlyNum(friend)

#subset RELIG (all rows and only the RELIG columns)
relig = data.iloc[:,414:422]
relig = onlyNum(relig)

#subset SOUL (all rows and only the SOUL columns. This is an adjancency matrix)
soul = data.iloc[:,422:501]
soul = onlyNum(soul)

#subset PLACE (all rows and only the PLACE columns)
place = data.iloc[:,501:536]
place = onlyNum(place)

#subset PROVIDE (all rows and only the PROVIDE columns)
provide = data.iloc[:,536:540]
provide = onlyNum(provide)

#subset MEET (all rows and only the MEET columns)
meet = data.iloc[:,540:560]
meet = onlyNum(meet)

#subset ATTRIBUTE (all rows and only the ATTRIBUTE columns such as education level, nationality, etc..)
attr = data.iloc[:,560:568]

'''
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
GRAPHING
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
'''

#Build a graph from the pandas subset that is an adjancency matrix
g = nx.convert_matrix.from_pandas_adjacency(commun)

#Find the biggest cluster and remove the unattached clusters
big_g = max(nx.connected_component_subgraphs(g), key=len)

#Since unconnected clusters were removed, we need to see what nodes are missing.
#Pull the nodes and dataframe index into a 'set'.  Subtract little set from the 
#big one to see what nodes are missing.
n = set(list(big_g.nodes))
i = set(list(attr['EDUC'].index))

#Get the missing nodes
missing = list(i-n)
missing[:] = [x -1 for x in missing]

#Remove the nodes from the dataframe by dropping the matching index
educ = attr['EDUC'].drop(attr['EDUC'].index[missing])

#It is a series.  Convert it back to dataframe
educ = educ.to_frame()
 
ColorLegend = {'Unknown': 0,'Elementary': 1,'Pesantren': 2,'High School': 3,
               'Some Univ': 4,'BA/BS': 5,'Some Grad': 6,'Masters': 7, 'PhD': 8}

pos = nx.spring_layout(big_g)

educ.loc[educ['EDUC'] == 0, 'labels'] = 'Unknown'
educ.loc[educ['EDUC'] == 1, 'labels'] = 'Elementary'
educ.loc[educ['EDUC'] == 2, 'labels'] = 'Pesantren'
educ.loc[educ['EDUC'] == 3, 'labels'] = 'High School'
educ.loc[educ['EDUC'] == 4, 'labels'] = 'Some Univ'
educ.loc[educ['EDUC'] == 5, 'labels'] = 'BA/BS'
educ.loc[educ['EDUC'] == 6, 'labels'] = 'Some Grad'
educ.loc[educ['EDUC'] == 7, 'labels'] = 'Masters'
educ.loc[educ['EDUC'] == 8, 'labels'] = 'PhD'


plt.figure(3, figsize=(12, 12))
nx.draw_networkx(big_g, node_color=educ['EDUC'], cmap=plt.cm.Set1, node_size=50)
plt.show()













