# EX-01 Implementation of Bayesian Networks

### Aim :
To create a bayesian Network for the given dataset in Python. &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**DATE: 00.00.2024**
### Algorithm:
Step 1:Import necessary libraries: pandas, networkx, matplotlib.pyplot, Bbn, Edge, EdgeType, BbnNode, Variable, EvidenceBuilder, InferenceController<br/>
Step 2:Set pandas options to display more columns<br/>
Step 3:Read in weather data from a CSV file using pandas<br/>
Step 4:Remove records where the target variable RainTomorrow has missing values<br/>
Step 5:Fill in missing values in other columns with the column mean<br/>
Step 6:Create bands for variables that will be used in the model (Humidity9amCat, Humidity3pmCat, and WindGustSpeedCat)<br/>
Step 7:Define a function to calculate probability distributions, which go into the Bayesian Belief Network (BBN)<br/>
Step 8:Create BbnNode objects for Humidity9amCat, Humidity3pmCat, WindGustSpeedCat, and RainTomorrow, using the probs() function to calculate their probabilities<br/>
Step 9:Create a Bbn object and add the BbnNode objects to it, along with edges between the nodes<br/>
Step 10:Convert the BBN to a join tree using the InferenceController<br/>
Step 11:Set node positions for the graph<br/>
Step 12:Set options for the graph appearance<br/>
Step 13:Generate the graph using networkx<br/>
Step 14:Update margins and display the graph using matplotlib.pyplot<br/>

### Program:
**Developed By: VIjayaraj V - 212222230174**
##### Import the necessary Libaries
```Python
import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
pd.options.display.max_columns=50
```
##### Read the Dataset
```Python
df=pd.read_csv('weatherAUS.csv', encoding='utf-8')
df=df[pd.isnull(df['RainTomorrow'])==False]
```
##### Data Preprocessing
```Python
df = df.drop(columns='Date')
numeric_columns = df.select_dtypes(include=['number']).columns
df.loc[:, numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
```
##### Create bands for variables that we want to use in the model
```Python
df['WindGustSpeedCat']=df['WindGustSpeed'].apply(lambda x: '0.<=40' if x<=40 else 
                                                 '1.40-50' if 40<x<=50 else '2.>50')
df['Humidity9amCat']=df['Humidity9am'].apply(lambda x: '1.>60' if x>60 else '0.<=60')
df['Humidity3pmCat']=df['Humidity3pm'].apply(lambda x: '1.>60' if x>60 else '0.<=60')
print(df)
```
##### calculate probability distribution, which goes into BBN (note, can handle up to 2 parents)
```Python
def probs(data, child, parent1=None, parent2=None):
    if parent1==None:
        prob=pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index().to_numpy().reshape(-1).tolist()
    elif parent1!=None:
            if parent2==None:
                prob=pd.crosstab(data[parent1],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
            else:
                prob=pd.crosstab([data[parent1],data[parent2]],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else: print("Error in Probability Frequency Calculations")
    return prob
```
##### Create nodes by using our earlier function to automatically calculate probabilities
```Python
H9am = BbnNode(Variable(0, 'H9am', ['<=60', '>60']), probs(df, child='Humidity9amCat'))
H3pm = BbnNode(Variable(1, 'H3pm', ['<=60', '>60']), probs(df, child='Humidity3pmCat', parent1='Humidity9amCat'))
W = BbnNode(Variable(2, 'W', ['<=40', '40-50', '>50']), probs(df, child='WindGustSpeedCat'))
RT = BbnNode(Variable(3, 'RT', ['No', 'Yes']), probs(df, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat'))
```
##### Create Network
```Python
bbn = Bbn() \
    .add_node(H9am) \
    .add_node(H3pm) \
    .add_node(W) \
    .add_node(RT) \
    .add_edge(Edge(H9am, H3pm, EdgeType.DIRECTED)) \
    .add_edge(Edge(H3pm, RT, EdgeType.DIRECTED)) \
    .add_edge(Edge(W, RT, EdgeType.DIRECTED))
```
##### Convert BBN to join tree and set node positions
```Python
join_tree = InferenceController.apply(bbn)
pos = {0: (-1, 2), 1: (-1, 0.5), 2: (1, 0.5), 3: (0, -1)}
```
##### Set options for graph looks
```Python
options = {"font_size": 16,"node_size": 4000, 
    "node_color": "white","edgecolors": "black",
    "edge_color": "red","linewidths": 5,"width": 5,}
```
##### Generate Graph,Update Margins and print graph
```Python
n, d = bbn.to_nx_graph()
nx.draw(n, with_labels=True, labels=d, pos=pos, **options)
ax = plt.gca()
ax.margins(0.10)
plt.axis("off")
plt.show()
print(probs(df, child='Humidity9amCat'))
print(probs(df, child='Humidity3pmCat', parent1='Humidity9amCat'))
print(probs(df, child='WindGustSpeedCat'))
print(probs(df, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat'))
```
### Output:
<img height=30% src="https://github.com/user-attachments/assets/ea0154f2-ad5f-4b89-811c-3d9a43bc2410">

### Result:
Thus a Bayesian Network is generated using Python.

