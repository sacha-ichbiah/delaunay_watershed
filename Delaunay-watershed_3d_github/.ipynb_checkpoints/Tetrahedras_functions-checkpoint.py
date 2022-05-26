import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy import interpolate
import imageio as io
import networkx 
import matplotlib.lines as lines

def give_faces_table(Tetrahedrons):
    #gives a table with a,b,c,ind_tetraedron
    #Tetrahedrons = tri.simplices.copy()
    #Tetrahedrons.sort(axis=1)
    Faces_table =[]
    for i,tet in enumerate(Tetrahedrons): 
        a,b,c,d = tet
        Faces_table.append([a,b,c,i])
        Faces_table.append([a,b,d,i])
        Faces_table.append([a,c,d,i])
        Faces_table.append([b,c,d,i])
    return(Faces_table)

def find_key_multiplier(num_points): 
    key_multiplier = 1
    while num_points//key_multiplier != 0 : 
        key_multiplier*=10
    return(key_multiplier)      
def lambdaSort_tri(Table,key_multiplier):
    Table.sort(key=lambda x:x[0]*(key_multiplier**3) + x[1]*(key_multiplier**2) + x[2]*(key_multiplier**1) + x[3])


"""
def order_relation_edges_table(tet_1,tet_2): 
    a_1,b_1,c_1,d_1=tet_1
    a_2,b_2,c_2,d_2=tet_2
    if a_1 > a_2 : 
        return 'sup'
    elif a_2 >a_1 : 
        return 'inf'
    else : 
        if b_1 > b_2 : 
            return 'sup'
        elif b_2 > b_1 :
            return 'inf'
        else : 
            if c_1 > c_2 : 
                return 'sup'
            elif c_2 > c_1 : 
                return 'inf'
            else : 
                if d_1>d_2 : 
                    return 'sup'
                elif d_2>d_1 : 
                    return 'inf'
                else : 
                    return 'equal'

def partition_tri(arr, low, high): 
    i = (low-1)         # index of smaller element 
    pivot = arr[high]     # pivot 

    for j in range(low, high): 

        # If current element is smaller than or 
        # equal to pivot 
        if order_relation_edges_table(arr[j],pivot)!='sup' : 

            # increment index of smaller element 
            i = i+1
            interm = arr[i].copy()
            arr[i]  = arr[j].copy()
            arr[j]  = interm.copy()
            
    interm = arr[i+1].copy()
    arr[i+1] = arr[high].copy()
    arr[high] = interm.copy()
    return (i+1) 

  
def quickSort_tri(arr, low, high): 
    if len(arr) == 1: 
        return arr 
    if low < high: 

        # pi is partitioning index, arr[p] is now 
        # at right place 
        pi = partition_tri(arr, low, high) 

        # Separately sort elements before 
        # partition and after partition 
        quickSort_tri(arr, low, pi-1) 
        quickSort_tri(arr, pi+1, high) 
"""




def construct_nodes_edges_list(Graph): 
    Nodes =np.zeros((len(Graph.Tetra),4),dtype=int)
    Indexes = np.zeros(len(Graph.Tetra),dtype=int)

    for i,edge in enumerate(Graph.Edges) : 
        a,b= edge.n1,edge.n2 
        Nodes[a,Indexes[a]]=i+1
        Nodes[b,Indexes[b]]=i+1
        Indexes[a]+=1
        Indexes[b]+=1

    return(Nodes)

def find_edges_highest_and_lowest(Graph):
    S=np.hstack(([-100],Graph.Scores))
    T=construct_nodes_edges_list(Graph)
    scores_node = S[T]
    Args=np.argsort(scores_node,axis=1)
    Args_min = Args[:,0]
    Args_max = Args[:,-1]
    Edges_highest = np.unique([T[idx,Args_max[idx]] for idx in range(len(T))])
    
    Times_min = np.zeros(len(Graph.Edges))
    Edges_min = [T[idx,Args_min[idx]] for idx in range(len(T))]
    for x in Edges_min : 
        Times_min[x]+=1
    Bool=(Times_min>=2).astype(int)
    Edges_double_min = np.nonzero(Bool)[0]
    return(Edges_highest,Edges_double_min)
















