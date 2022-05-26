import networkx
import numpy as np
from time import time

#####
#FILTERING
#####

def Clean_map_volume(List_Graphs,List_Maps, vol_min,vol_max): 
    graph = List_Graphs[-1]
    Map = List_Maps[-1]
    n_nodes = len(graph.nodes)
    
    Filtered_node_list = []
    for node_elmt in list(graph.nodes.data()) : 
        vol = node_elmt[1]['volume']
        if vol > vol_min and vol < vol_max : 
            Filtered_node_list.append(node_elmt)

            
    indexes_kept = [x[0] for x in Filtered_node_list]
    volumes_dict = [x[1] for x in Filtered_node_list]

    n_nodes = len(graph.nodes)
    filtering_table = np.zeros(n_nodes,dtype = np.int)
    filtering_table[indexes_kept]+=1
    Filtered_edge_list = []
    Indexes = np.delete(np.arange(n_nodes),np.nonzero(1-filtering_table))
    Map_end = dict(zip(np.arange(len(Indexes)),[{x} for x in Indexes]))
    reverse_map = Reverse_map(Map_end)


    for edge in graph.edges.data() : 
        n1, n2,inf = edge
        if filtering_table[n1]==1 and filtering_table[n2]==1 : 
            Filtered_edge_list.append((reverse_map[n1],reverse_map[n2],inf))
            
    G = networkx.Graph()
    
    G.add_nodes_from(zip(np.arange(len(Filtered_node_list)),volumes_dict))
    G.add_edges_from(Filtered_edge_list)
    
    List_Graphs.append(G)
    List_Maps.append(Map_end)




#####
#GENERAL
#PURPOSE
#FUNCTIONS
#####

def Reverse_map(Map): 
    #The reverse map is a dictionnary where reverse_map[ind]=connected_component to which belongs the ind
    reverse_map = {}
    for key in Map:
        for a in Map[key]: 
            reverse_map[a]=key
    return(reverse_map)


def Reverse_map_old(Map): 
    #The reverse map is a dictionnary where reverse_map[ind]=connected_component to which belongs the ind
    reverse_map = {}
    for i in range(len(Map)) : 
        for a in Map[i]: 
            reverse_map[a]=i
    return(reverse_map)


def combine_Maps(Map1,Map2):
    #The operation from Map2 has been done after the operation from Map1
    Map_comb={}
    for cluster_num in Map2 :
        Map_comb[cluster_num]=[]
        for num in Map2[cluster_num]: 
            Map_comb[cluster_num]+=list(Map1[num])
    return(Map_comb)

def combine_Map_list(Maps): 
    if len(Maps)>1 : 
        Map_comb = Maps[-1]
        for i in range(len(Maps)-2,-1,-1): 
            Map_comb = combine_Maps(Maps[i],Map_comb)
    else : 
        Map_comb = dict(zip(Maps[0].keys(),[list(Maps[0][key]) for key in Maps[0].keys()]))
    return(Map_comb)

def Assemble_Edges(Graph): 
    #See the definition of the different attributes in the method networkx_graph_weights_and_borders of the class Graph
    New_Graph = networkx.Graph()
    Edges=[]
    for node in Graph.nodes : 
        for neighbor in Graph.neighbors(node): 
            if node<neighbor : 
                
                weight=0
                area = 0
                
                num_edges= Graph.number_of_edges(node,neighbor)

                for key in range(num_edges): 
                    
                    area += Graph[node][neighbor][key]['area']
                    weight += Graph[node][neighbor][key]['score']*Graph[node][neighbor][key]['area']

                score = weight/area
                Edges.append((node,neighbor,{'score': score,'area':area}))
            
    New_Graph.add_nodes_from(Graph.nodes.data())
    New_Graph.add_edges_from(Edges)
    return(New_Graph)

#####
#SEEDED WATERSHED
#####
def Seeded_Watershed(List_Graphs,List_Maps,seeds,indices_labels,zero_nodes = []): 
    Graph = List_Graphs[-1]
    Map = seeded_watershed_map(Graph,seeds,indices_labels,zero_nodes)

    reverse_map = Reverse_map(Map)
    Edges=[]
    
    Initial_nodes_volumes = Graph.nodes.data('volume')
    Cluster_volumes=[]

    for key in Map:
        nodes_subset = list(Map[key])
        cluster_volume = 0
        for node_idx in nodes_subset : 
             
            cluster_volume+=Initial_nodes_volumes[node_idx]
            
            for edge in Graph.edges(node_idx,data=True) : 
                ind_1 = reverse_map[edge[0]]
                ind_2 = reverse_map[edge[1]]
                if ind_1<ind_2 :
                    Edges.append((ind_1,ind_2,edge[2]))

        Cluster_volumes.append(cluster_volume)
            
    volume_dicts = list(map(lambda x:{'volume':x},Cluster_volumes))
    Nodes_table = zip(list(Map.keys()),volume_dicts)

    Aggregated_Graph = networkx.MultiGraph()
    Aggregated_Graph.add_edges_from(Edges)
    Aggregated_Graph.add_nodes_from(Nodes_table)
    Aggregated_Graph.remove_edges_from(list(networkx.selfloop_edges(Aggregated_Graph)))
    Aggregated_Graph=Assemble_Edges(Aggregated_Graph)
    List_Graphs.append(Aggregated_Graph)
    List_Maps.append(Map)

def seeded_watershed_aggregation(Graph,seeds,indices_labels): 
    #Seeds are expressed as labels of the nodes
    Labels = np.zeros(len(Graph.nodes),dtype=int)-1

    for i,seed in enumerate(seeds) : 
        Labels[seed]=indices_labels[i]
        
    Groups={}
    Number_Group=np.zeros(len(Graph.nodes),dtype=int)-1
    num_group = 0
    
    args = np.argsort(-np.array(list(Graph.edges.data('score')))[:,2])
    Edges = list(Graph.edges)
    for arg in args: 
        a,b = Edges[arg]
        if Labels[a]!=-1 and Labels[b]!=-1 : 
            continue
        elif Labels[a]!=-1 and Labels[b]==-1 : 
            group = Groups.get(Number_Group[b],[b])
            Labels[group]=Labels[a]
        elif Labels[b]!=-1 and Labels[a]==-1 : 
            group = Groups.get(Number_Group[a],[a])
            Labels[group]=Labels[b]
        else : 
            if Number_Group[a]!=-1 : 
                if Number_Group[a]==Number_Group[b] : 
                    continue
                elif Number_Group[b]!=-1 : 
                    old_b_group = Groups.pop(Number_Group[b])
                    Groups[Number_Group[a]]+=old_b_group
                    Number_Group[old_b_group]=Number_Group[a]
                else : 
                    Groups[Number_Group[a]].append(b)
                    Number_Group[b]=Number_Group[a]
            else : 
                if Number_Group[b]!=-1 : 
                    Groups[Number_Group[b]].append(a)
                    Number_Group[a]=Number_Group[b]
                else : 
                    Number_Group[a]=num_group
                    Number_Group[b]=num_group
                    Groups[num_group]=[a,b]
                    num_group+=1
    return(Labels)

def seeded_watershed_map(Graph,seeds,indices_labels,zero_nodes = []): 
    Labels = seeded_watershed_aggregation(Graph,seeds,indices_labels)
    Labels[zero_nodes]=0
    for i,seed in enumerate(seeds) : 
        Labels[seed]=indices_labels[i]
        
    Map_end = build_Map_from_labels(Labels)
    print("Number of labels",np.unique(Labels),len(np.unique(Labels)))
    return(Map_end)


def build_Map_from_labels(PHI): 
    Map_end ={}
    for idx, label in enumerate(PHI) : 
        Map_end[label] = Map_end.get(label,[])
        Map_end[label].append(idx)
    return(Map_end)









#######
#Sorted aggregation 
#######

#Operation to do after a watershed algorithm : The result is prone to be a bit oversegmented. We fuse the different instances using the sorted_aggregation algorithm. 
#Very slow operation
def Sorted_aggregation(List_Graphs,List_Maps,lambda_func): 
    #Differs from the DCUT aggregation from the fact that now the edges are examined one by one from the biggest to the lowest. 
    Graph = List_Graphs[-1].copy()
    print(len(Graph.nodes.data()))
    Map=dict(zip(range(len(Graph.nodes)),[[i] for i in range(len(Graph.nodes))]))
    nodes_found = True #indicates if we have found two nodes to fuse or not (it is the termination condition)
    while nodes_found : 
        nodes_found = False
        Edges = np.array(list(Graph.edges.data()))
        Args = np.argsort(-(np.array(list(Graph.edges.data('score')))[:,2]))
        
        #Most of the time we will only do a single iteration through this loop
        for arg in Args: 
            node_1,node_2, info = Edges[arg]
            if not lambda_func(info): 
                nodes_found = True
                break
        
        if nodes_found : 
            node_min = min(node_1,node_2)
            node_max = max(node_1,node_2)
            Graph = contract_nodes(Graph,node_min,node_max)
            Map[node_min]=Map[node_min]+Map.pop(node_max)

    List_Graphs.append(Graph)
    Map_end = dict(zip(np.arange(len(Map)),[v for u,v in Map.items()]))
    List_Maps.append(Map_end)


def contract_nodes(Graph,node_1,node_2):
    New_Graph = Graph.copy()
    New_Graph.remove_node(node_1)
    New_Graph.remove_node(node_2)

    Edges = list(New_Graph.edges.data())
    for edge in Graph.edges(node_1,data=True) : 
        ind_1, ind_2, info = edge
        if ind_2 != node_2 : 
            Edges.append((node_1, ind_2, info))
    for edge in Graph.edges(node_2,data=True) : 
        ind_1, ind_2, info = edge
        if ind_2 != node_1 : 
            Edges.append((node_1, ind_2,info))
    
    New_Graph.add_node(node_1,volume = (Graph.nodes[node_1]['volume']+Graph.nodes[node_2]['volume']))
    Aggregated_Graph = networkx.MultiGraph()
    Aggregated_Graph.add_nodes_from(New_Graph.nodes.data())
    Aggregated_Graph.add_edges_from(Edges)
    Aggregated_Graph=Assemble_Edges(Aggregated_Graph)
    return(Aggregated_Graph)

