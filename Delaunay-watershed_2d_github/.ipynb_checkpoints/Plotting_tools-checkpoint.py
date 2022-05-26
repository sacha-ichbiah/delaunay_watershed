import numpy as np 
import matplotlib.pyplot as plt
import networkx
import imageio as io
from skimage.feature import peak_local_max
from scipy.spatial import ckdtree
from scipy.interpolate import RegularGridInterpolator
from matplotlib import cm

def retrieve_border_lines(Delaunay_Graph,Map):
    reverse_map ={}
    for key in Map : 
         for node_idx in Map[key] :
             reverse_map[node_idx]=key
    Clusters=[]
    for i in range(len(Map)): 
        Clusters.append([])
        
    for idx in range(len(Delaunay_Graph.Edges)) : 
        edge = Delaunay_Graph.Edges[idx]
        nodes_linked = Delaunay_Graph.Nodes_linked_by_Edges[idx]
        cluster_1 = reverse_map[nodes_linked[0]]
        cluster_2 =reverse_map[nodes_linked[1]]
        #if the two nodes of the edges belong to the same cluster we ignore them
        #otherwise we add them to the polygonal lines dict
        if cluster_1 != cluster_2 : 
            v1,v2=edge[0],edge[1]
            Clusters[cluster_1].append([v1,v2])
            Clusters[cluster_2].append([v1,v2])   
    
    for idx in range(len(Delaunay_Graph.Lone_edges)):
        edge = Delaunay_Graph.Lone_edges[idx]
        node_linked = Delaunay_Graph.Nodes_linked_by_lone_edges[idx]
        cluster_1 = reverse_map[node_linked]
        #We incorporate all these edges because they are border edges
        if cluster_1 !=0:
            v1,v2=edge[0],edge[1]
            Clusters[cluster_1].append([v1,v2])

    return(Clusters)

def retrieve_border_lines_old(Delaunay_Graph,Map):
    reverse_map ={}
    for key in Map : 
         for node_idx in Map[key] :
             reverse_map[node_idx]=key
    Clusters=[]
    for i in range(len(Map)): 
        Clusters.append([])
        
    for idx in range(len(Delaunay_Graph.Edges)) : 
        edge = Delaunay_Graph.Edges[idx]
        nodes_linked = Delaunay_Graph.Nodes_linked_by_Edges[idx]
        cluster_1 = reverse_map[nodes_linked[0]]
        cluster_2 =reverse_map[nodes_linked[1]]
        #if the two nodes of the edges belong to the same cluster we ignore them
        #otherwise we add them to the polygonal lines dict
        if cluster_1 != cluster_2 : 
            v1,v2=edge[0],edge[1]
            Clusters[cluster_1].append([v1,v2])
            Clusters[cluster_2].append([v1,v2])     
    return(Clusters)

def plot_line(line,Verts,color=np.random.rand(3)):
    for elmts in line : 
        a,b,=elmts
        coords = Verts[[a,b]]
        plt.plot(coords[:,1],coords[:,0],'o-',color=color,linewidth=2, markersize=2)

def plot_line_no_marker(line,Verts,color=np.random.rand(3)):
    for elmts in line : 
        a,b,=elmts
        coords = Verts[[a,b]]
        plt.plot(coords[:,1],coords[:,0],'-',color=color,linewidth=2)
        
def plot_lines(Lines,Verts,random_seed=0): 
    np.random.seed(random_seed)
    for line in Lines : 
        color=np.random.rand(3)
        plot_line(line, Verts,color=color)

def plot_polylines_no_marker(Lines,Verts,figsize=15,random_seed=0): 
    np.random.seed(random_seed)
    for line in Lines : 
        color=np.random.rand(3)
        plot_line_no_marker(line, Verts,color=color)

def plot_polylines(Lines,Verts,figsize=15,random_seed=0): 
    np.random.seed(random_seed)
    for line in Lines : 
        color=np.random.rand(3)
        plot_line(line, Verts,color=color)
        
def plot_line_ax(ax,line,Verts,color=np.random.rand(3)):
    for elmts in line : 
        a,b,=elmts
        coords = Verts[[a,b]]
        ax.plot(coords[:,1],coords[:,0],'o-',color=color,linewidth=1, markersize=3)

def plot_line_no_marker_ax(ax,line,Verts,color=np.random.rand(3)):
    for elmts in line : 
        a,b,=elmts
        coords = Verts[[a,b]]
        ax.plot(coords[:,1],coords[:,0],'-',color=color,linewidth=2)
        
def plot_lines_ax(ax,Lines,Verts,random_seed=0): 
    np.random.seed(random_seed)
    for line in Lines : 
        color=np.random.rand(3)
        plot_line_ax(ax,line, Verts,color=color)
    ax.set_aspect('equal', adjustable='box')

def plot_polylines_no_marker_ax(ax,Lines,Verts,figsize=15,random_seed=0): 
    np.random.seed(random_seed)
    for line in Lines : 
        color=np.random.rand(3)
        plot_line_no_marker_ax(ax,line, Verts,color=color)
    ax.set_aspect('equal', adjustable='box')

def plot_polylines_ax(ax,Lines,Verts,figsize=15,random_seed=0): 
    np.random.seed(random_seed)
    for line in Lines : 
        color=np.random.rand(3)
        plot_line_ax(ax,line, Verts,color=color)
    ax.set_aspect('equal', adjustable='box')

def compute_barycentres(Map_end,Delaunay_Graph): 
    Triangles = Delaunay_Graph.tri.simplices
    elmts = [v for u,v in Map_end.items()]
    Barycentres = np.array([np.mean(Delaunay_Graph.Vertices[np.unique(Triangles[v].flatten())],axis=0) for v in elmts])
    New_Barycentres = Barycentres.copy()
    New_Barycentres[:,0]=Barycentres[:,1]
    New_Barycentres[:,1]=1-Barycentres[:,0]
    return(New_Barycentres)



###
##NEW FUNCTIONS
###
###
###




def create_figure(figsize,alpha): 
    fig, ax = plt.subplots(figsize=(figsize,figsize*alpha))
    ax.set_aspect('equal')
    return(fig)

def plot_image(Image): 
    plt.imshow(Image,cmap=plt.cm.gray)


def plot_edges(Delaunay_Graph): 
    for i,edge in enumerate(Delaunay_Graph.Edges) : 
        plt.plot([Delaunay_Graph.Vertices[edge[0]][1], Delaunay_Graph.Vertices[edge[1]][1]],[Delaunay_Graph.Vertices[edge[0]][0], Delaunay_Graph.Vertices[edge[1]][0]],'k',linewidth=2)
        
def plot_triangulation_ax(ax,Seg): 
    tesselation = Seg.tri
    ax.triplot(tesselation.points[:,1],tesselation.points[:,0], tesselation.simplices,color='black')
    ax.scatter(tesselation.points[:,1],tesselation.points[:,0],color='red')


def plot_triangles_coloured(Delaunay_Graph,Map,random_seed = 1,colors='rand'): 
    Verts = Delaunay_Graph.Vertices.copy()[:,[1,0]]
    Verts[:,1]=np.amax(Verts[:,1])-Verts[:,1]
    np.random.seed(random_seed)
    Tris = (Verts[Delaunay_Graph.tri.simplices])
    maximum = max(Map.keys())
    for key in Map : 
        if colors == 'viridis':
            color = cm.viridis(key/maximum)
        else : 
            color = np.random.rand(3)
        for node_idx in Map[key] :
            tri = Tris[node_idx]
            t = plt.Polygon(tri,color=color)
            plt.gca().add_patch(t)
            
    plt.xlim(0,np.amax(Verts[:,0]))
    plt.ylim(0,np.amax(Verts[:,1]))
    plt.gca().set_aspect('equal', adjustable='box')
    return(plt.gca())

def plot_triangles_coloured_ax(ax,Delaunay_Graph,Map,random_seed = 1,colors='rand'): 
    Verts = Delaunay_Graph.Vertices.copy()[:,[1,0]]
    Verts[:,1]=np.amax(Verts[:,1])-Verts[:,1]
    np.random.seed(random_seed)
    Tris = (Verts[Delaunay_Graph.tri.simplices])
    maximum = max(Map.keys())
    for key in Map : 
        if key ==0 : 
            color = np.zeros(3)
        elif colors == 'viridis':
            color = cm.viridis(key/maximum)
        else : 
            color = np.random.rand(3)
        for node_idx in Map[key] :
            tri = Tris[node_idx]
            t = plt.Polygon(tri,color=color)
            ax.add_patch(t)

    ax.set_xlim(0,np.amax(Verts[:,0]))
    ax.set_ylim(0,np.amax(Verts[:,1]))
    ax.set_aspect('equal', adjustable='box')


def create_coords(nx,ny):
    XV = np.linspace(0,nx-1,nx)
    YV = np.linspace(0,ny-1,ny)
    xvv, yvv = np.meshgrid(XV,YV )
    xvv=xvv.transpose().flatten()
    yvv=yvv.transpose().flatten()
    Points=np.transpose(np.stack(([xvv,yvv])))
    return(Points)

def compute_seeds_idx_from_pixel_coords(EDT,Centroids,Coords,plot_figure=True):
    ##########
    ## Compute the seeds used for watershed
    ##########
    
    nx,ny = EDT.shape
    Points = create_coords(nx,ny)

    Anchors = Coords[:,0]*ny+Coords[:,1]
    p=Points[Anchors]

    if plot_figure : 
        plt.figure(figsize=(8,8))
        plt.imshow(EDT)
        plt.scatter(p[:,1],p[:,0],color='r')
    tree = ckdtree.cKDTree(Centroids)
    Dist,Idx_seeds = tree.query(p)
    unique,indices = np.unique(Idx_seeds,return_index=True)
    return(Idx_seeds[sorted(indices)])

def compute_coords_local_min(DT,min_distance,plot_figure=True):
    nx,ny = DT.shape
    alpha = ny/nx
    Points = create_coords_not_square(nx,ny)
    Local_maxes = peak_local_max(DT, min_distance=min_distance)
    Anchors_bis=Local_maxes[:,0]*ny+Local_maxes[:,1]
    Points_local_max = Points[Anchors_bis]
    if plot_figure : 
        plt.imshow(DT,extent=(0.,alpha,0.,1))
        plt.scatter(Points_local_max[:,1],1-Points_local_max[:,0],color='r')
    return(Local_maxes)