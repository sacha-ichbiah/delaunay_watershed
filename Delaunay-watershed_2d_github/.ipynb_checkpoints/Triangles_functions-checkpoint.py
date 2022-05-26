import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as lines

def give_edges_table(Faces):
    #gives a table with a,b,ind_face

    Edges_table =[]
    for i,face in enumerate(Faces): 
        a,b,c = face
        Edges_table.append([a,b,i])
        Edges_table.append([a,c,i])
        Edges_table.append([b,c,i])
    return(Edges_table)

def order_relation_edges_table(f_1,f_2): 
    a_1,b_1,c_1=f_1
    a_2,b_2,c_2=f_2
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

def find_key_multiplier(num_points): 
    key_multiplier = 1
    while num_points//key_multiplier != 0 : 
        key_multiplier*=10
    return(key_multiplier)

def lambdaSort_tri(Table,key_multiplier): 
    Table.sort(key=lambda x:x[0]*(key_multiplier**2) + x[1]*(key_multiplier**1) + x[0])

        
class Triangle():
    def __init__(self):
        #Edges is a table of edges
        self.Edges = []

   

def min_pool(x,kernel = 3, stride = 1): 
    pool = torch.nn.MaxPool2d(kernel_size=kernel,stride=stride,return_indices=True)
    Values, Indices = pool(-x)
    Values*=-1
    return(Values, Indices)











