import numpy as np
from scipy.spatial import ckdtree
from skimage.feature import peak_local_max
from Non_square_EDT import *
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Delaunay
from skimage.filters import gaussian
import matplotlib.pyplot as plt

def Adjust_distance_points_scipy_interp(Anchor_Points,f,threshold):
    points = Anchor_Points.copy()
    edt = np.array(f(points))
    args = np.argsort(edt)
    points=points[args]
    edt=edt[args]

    print("Initial number of points :",len(points))
    #edt is a table containing the edt of all the points
    #we need to remove the points for this table too to keep the correspondance between the indices
    out_condition=False
    while not out_condition : 
        tree=ckdtree.cKDTree(points)
        n_points_in = len(points)
        index = 0
        points_to_remove=[]
        points_removed=np.zeros(len(points))
        while index<len(points): 
            #First : if the point is already removed we do not remove its neighbors
            if points_removed[index]==1 : 
                index+=1 
                continue

            #NN search in a certain radius
            N = tree.query_ball_point(points[index],threshold)

            #If only one neighbor : its itself -> next
            if len(N)<=1 : 
                index+=1

            #Else with more neighbors : 
            else : 
                #If there are neighbors with lower edt the point is already remove so we would never get to this stage. 
                for idx in N : 
                    if idx !=index : 
                        points_to_remove.append(idx)
                        points_removed[idx]=1
                index+=1
        points = np.delete(points,points_to_remove,axis=0)
        edt = np.delete(edt,points_to_remove,axis=0)

        n_points_out = len(points)
        if n_points_in==n_points_out : 
            out_condition=True


    print("Final number of points:",len(points))
    return(points)

"""
def give_anchor_points(DT,values_thresh,distance_thresh,min_dist=10,add_local_min_max=True):
    nx, ny = DT.shape
    alpha=ny/nx

    Indices = np.arange(nx*ny).reshape(DT.shape)
    Anchors = Indices[DT<values_thresh].flatten()

    Points = create_coords_torch_not_square(nx,ny)
    square_anchors = give_anchors_not_square(1,alpha)
    Points_delaunay = torch.cat([square_anchors,Points[Anchors]])

    Local_maxes = peak_local_max(-DT, min_distance=min_dist)
    Anchors_bis=Local_maxes[:,0]*ny+Local_maxes[:,1]
    Points_local_min = Points[Anchors_bis]

    Local_maxes = peak_local_max(DT, min_distance=min_dist)
    Anchors_bis=Local_maxes[:,0]*ny+Local_maxes[:,1]
    Points_local_max = Points[Anchors_bis]

    points = np.vstack((Points_delaunay))
    if add_local_min_max : 
        points = np.vstack((points,Points_local_max))
        points = np.vstack((points,Points_local_min))

    x = np.linspace(0,alpha,DT.shape[1])
    y = np.linspace(0,1,DT.shape[0])
    h = RegularGridInterpolator((y, x) ,DT)

    points=Adjust_distance_points_scipy_interp(points,h,threshold=distance_thresh)
    
    return(points,Points_local_min,Points_local_max)
"""

def give_pixel_size(shape): 
    return(1/np.amin(shape))


def create_coords_ns(nx,ny):
    XV = np.linspace(0,nx-1,nx).astype(int)
    YV = np.linspace(0,ny-1,ny).astype(int)
    xvv, yvv = np.meshgrid(XV,YV )
    xvv=xvv.transpose().flatten()
    yvv=yvv.transpose().flatten()
    Points=np.transpose(np.stack(([xvv,yvv])))
    return(Points)

def find_lowest_point(x0, y0, x1, y1,Total_EDT):
    "Bresenham's line algorithm - modified from https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm#Python"
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append([x,y])
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append([x,y])
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy        
    points.append([x,y])
    
    func_total_edt = lambda point : Total_EDT[point[0],point[1]]
    edt_points = np.array(list(map(func_total_edt,points)))
    edt_diffs = edt_points[:-1]-edt_points[1:]

    return(points[np.sum(edt_diffs>0)])


def find_better_points(points,Total_EDT,mask_1,plot=False): 
    from scipy.spatial import cKDTree
    
    Points = create_coords_ns(Total_EDT.shape[0],Total_EDT.shape[1])
    points_ext = Points[mask_1.flatten()]
    
    Tree = cKDTree(points_ext)
    dist,indices = Tree.query(points)
    dir_grad = (points - points_ext[indices]).astype(float)
    norm = np.linalg.norm(dir_grad,axis=1)
    dir_grad/=norm.reshape(-1,1)
     
    func_total_edt = lambda x: Total_EDT[x[0],x[1]]
    points_end = (points +dir_grad*(np.array(list(map(func_total_edt,points))).reshape(-1,1))).astype(int)
    
    #diffs = line(points_in[0,0],points_in[0,1],points_end[0,0],points_end[0,1],Total_EDT)
    New_points = []
    for i in range(len(points)): 
        New_points.append(find_lowest_point(points[i,0],points[i,1],points_end[i,0],points_end[i,1],Total_EDT))
    New_points = np.array(New_points)
    #print(New_points)
    if plot : 
        
        plt.figure(figsize=(5,5))
        plt.scatter(points[:,1],1-points[:,0])
        plt.scatter(points_end[:,1],1-points_end[:,0])
        plt.scatter(New_points[:,1],1-New_points[:,0])
    return(New_points)

def give_anchor_points(DT,values_thresh,distance_thresh,min_dist=10):
    
    ##########
    ## Compute the anchor points used for the triangulation
    ## values_thresh defines the maximal value that a point can have to be defined as an anchor point
    ## distance_thresh is the minimal distance between two anchor points (in the box [0,1]*[0,alpha]) 
    ## min_dist is the minimal distance (in pixels) between two local minimas/maximas. 
    ## add_local_min_max is True if we add local minimas and maximas to the anchor points, otherwise they are not added
    ##########

    nx, ny = DT.shape
    alpha=ny/nx

    Indices = np.arange(nx*ny).reshape(DT.shape)
    Anchors = Indices[DT<values_thresh].flatten()

    Points = create_coords_ns(nx,ny)
    square_anchors = give_anchors_not_square(1,alpha)
    Points_delaunay = np.concatenate([square_anchors,Points[Anchors]])
    
    f_DT = gaussian(DT,sigma=1)
    Points_local_max = peak_local_max(f_DT, min_distance=min_dist)
    #Anchors_bis=Local_maxes[:,0]*ny+Local_maxes[:,1]
    #Points_local_max = Points[Anchors_bis]

    points = np.vstack((Points_delaunay))

    x = np.linspace(0,DT.shape[1]-1,DT.shape[1])
    y = np.linspace(0,DT.shape[0]-1,DT.shape[0])
    h = RegularGridInterpolator((y, x) ,DT)
    
    points=Adjust_distance_points_scipy_interp(points,h,threshold=distance_thresh)

    return(points,Points_local_max)
def double_filtering(Total_EDT,mask_1,threshold_on_EDT,distance_thresh,min_dist):
    distance_thresh = min_dist
    points,Points_local_max = give_anchor_points(Total_EDT,values_thresh=threshold_on_EDT,distance_thresh=distance_thresh,min_dist=min_dist)
    
    eps=1
    Indexes = (points[:,0]<eps).astype(int) + (points[:,1]<eps).astype(int) + (points[:,0]>Total_EDT.shape[0]-1-eps).astype(int) + (points[:,1]>Total_EDT.shape[1]-1-eps).astype(int)
    interior_points=points[Indexes==0].astype(int)
    exterior_points=points[Indexes>0]
    
    interior_points = find_better_points(interior_points,Total_EDT,mask_1,plot=False)
    
    x = np.linspace(0,Total_EDT.shape[1]-1,Total_EDT.shape[1])
    y = np.linspace(0,Total_EDT.shape[0]-1,Total_EDT.shape[0])
    h = RegularGridInterpolator((y, x) ,Total_EDT)
    
    interior_points = Adjust_distance_points_scipy_interp(interior_points,h,distance_thresh)#*np.sqrt(2)/2)
    
    points = np.vstack((interior_points,exterior_points,Points_local_max)).astype(int)
    idx_max_points = len(interior_points)
    return(points,idx_max_points)

def auto_double_edt(DT, val_thresh,normalize=True):
    from scipy import ndimage
    mask_1 = DT>val_thresh
    mask_2 = (1-DT)>(1-val_thresh)
    mask_1 = np.pad(mask_1, ((1, 1), (1, 1)), 'constant',constant_values = 0)
    mask_2 = np.pad(mask_2, ((1, 1), (1, 1)), 'constant',constant_values = 1)
    EDT_1 = ndimage.distance_transform_edt(mask_1)
    EDT_2 = ndimage.distance_transform_edt(mask_2)
    inv = np.amax(EDT_2)-EDT_2
    Total_EDT = (EDT_1+np.amax(EDT_2))*mask_1 + inv*mask_2
    #Total_EDT = skimage.filters.gaussian(Total_EDT,sigma=1)  BLUR IT OR NOT BLUR IT, THAT IS THE QUESTION
    vt = np.amax(EDT_2)
    if normalize : 
        Total_EDT/=np.amax(Total_EDT)
        vt/=np.amax(Total_EDT)
    
    return(Total_EDT,np.amax(EDT_2),vt)




