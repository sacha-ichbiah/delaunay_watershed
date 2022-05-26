import numpy as np
import networkx as nx
from mayavi import mlab
import matplotlib.pyplot as plt
import imageio as io
from skimage.feature import peak_local_max
from scipy.spatial import ckdtree
#import napari
from scipy.ndimage.measurements import center_of_mass, label
from Mesh_utilities import separate_faces,separate_faces_dict


import polyscope as ps

def retrieve_border_tetra_with_index_map(Graph,Map):
    reverse_map ={}
    for key in Map : 
            for node_idx in Map[key] :
                reverse_map[node_idx]=key
                
    Clusters=[]
    for _ in range(len(Map)): 
        Clusters.append([])

    for idx in range(len(Graph.Faces)) : 
        nodes_linked = Graph.Nodes_Linked_by_Faces[idx]

        cluster_1 = reverse_map.get(nodes_linked[0],-1)
        cluster_2 =reverse_map.get(nodes_linked[1],-2)
        #if the two nodes of the edges belong to the same cluster we ignore them
        #otherwise we add them to the mesh
        if cluster_1 != cluster_2 : 
            face = Graph.Faces[idx]
            if cluster_1 >= 0 : 
                Clusters[cluster_1].append(face)
            if cluster_2 >= 0 : 
                Clusters[cluster_2].append(face)
            

    for idx in range(len(Graph.Lone_Faces)):
        edge = Graph.Lone_Faces[idx]
        node_linked = Graph.Nodes_linked_by_lone_faces[idx]
        cluster_1 = reverse_map[node_linked]
        #We incorporate all these edges because they are border edges
        if cluster_1 !=0:
            v1,v2,v3=edge[0],edge[1],edge[2]
            Clusters[cluster_1].append([v1,v2,v3])
    return(Clusters)


def compute_seeds_idx_from_voxel_coords(EDT,Centroids,Coords):
    ##########
    ## Compute the seeds used for watershed
    ##########
    
    nx,ny,nz = EDT.shape
    Points = create_coords(nx,ny,nz)

    Anchors = Coords[:,0]*ny*nz+Coords[:,1]*nz+Coords[:,2]
    p=Points[Anchors]

    tree = ckdtree.cKDTree(Centroids)
    Dist,Idx_seeds = tree.query(p)
    return(Idx_seeds)
    #unique,indices = np.unique(Idx_seeds,return_index=True)
    #return(Idx_seeds[indices])
    #return(Idx_seeds[sorted(indices)])

def create_coords(nx,ny,nz):
    XV = np.linspace(0,nx-1,nx)
    YV = np.linspace(0,ny-1,ny)
    ZV = np.linspace(0,nz-1,nz)
    xvv, yvv, zvv = np.meshgrid(XV,YV,ZV)
    xvv=np.transpose(xvv,(1,0,2)).flatten()
    yvv=np.transpose(yvv,(1,0,2)).flatten()
    zvv=zvv.flatten()
    Points=np.vstack(([xvv,yvv,zvv])).transpose().astype(int)
    return(Points)






def plot_cells_polyscope(Verts,Faces,clean_before = True, clean_after = True):
    Clusters = separate_faces_dict(Faces)
    
    ps.init()
    
    ps.set_ground_plane_mode("none")
    if clean_before : 
        ps.remove_all_structures()
        
    for key in sorted(Clusters.keys()):
        cluster = Clusters[key]
        ps.register_surface_mesh("Cell "+str(key), Verts, np.array(cluster), smooth_shade=False)
    ps.show()
    
    if clean_after : 
        ps.remove_all_structures()
    

def compute_barycentres_from_clusters(Clusters,Graph): 
    Barycentres = np.array([np.mean(Graph.Vertices[np.unique(np.array(cluster,dtype=int).flatten())],axis=0) for cluster in Clusters])
    New_Barycentres = Barycentres.copy()
    #New_Barycentres[:,0]=Barycentres[:,1]
    #New_Barycentres[:,1]=1-Barycentres[:,0]
    return(New_Barycentres)




    


