import numpy as np
from scipy.spatial import ckdtree
from skimage.feature import peak_local_max
from scipy.ndimage.measurements import center_of_mass, label
from Non_square_EDT import *
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Delaunay
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import ndimage
from time import time
import skimage.filters

def Adjust_distance_points_scipy_interp(Anchor_Points,f,threshold):
    points = Anchor_Points.copy()
    edt = np.array(f(points))
    args = np.argsort(edt)
    points=points[args]
    edt=edt[args]

    #edt is a table containing the edt of all the points
    #we need to remove the points for this table too to keep the correspondance between the indices
    out_condition=False
    print("Number of points before filtering :",points.shape[0])
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
    print("Number of points after filtering :",points.shape[0])
    return(points)


def interpolate_map(DT):
    nx, ny,nz = DT.shape
    alpha = ny/nx
    beta = nz/nx
    x = np.linspace(0,1,DT.shape[0])
    y = np.linspace(0,alpha,DT.shape[1])
    z = np.linspace(0,beta,DT.shape[2])
    h = RegularGridInterpolator((x,y,z) ,DT)
    return(h)

def give_voxel_size(shape): 
    return(1/np.amin(shape))



def give_anchors_not_cube_big(cube_size,alpha,beta): 
    Points=np.zeros((8,3))
    index=0
    for i in range(2): 
        for j in range(2): 
            for k in range(2): 
                Points[index]=np.array([i,j,k])
                index+=1
    Points[:,0]*=cube_size-1
    Points[:,1]*=alpha*(cube_size-1)
    Points[:,2]*=beta*(cube_size-1)
    return(Points)

def give_anchors_not_cube(nx,ny,nz): 
    alpha = ny/nx
    beta = nz/nx
    Points=np.zeros((8,3))
    index=0
    for i in range(2): 
        for j in range(2): 
            for k in range(2): 
                Points[index]=np.array([i,j,k])
                index+=1
                
    Points[:,1]*=alpha
    Points[:,2]*=beta
    return(Points)

def create_coords_not_cube(nx,ny,nz):
    XV = np.linspace(0,nx-1,nx)
    YV = np.linspace(0,ny-1,ny)
    ZV = np.linspace(0,nz-1,nz)
    xvv, yvv, zvv = np.meshgrid(XV,YV,ZV)
    xvv=np.transpose(xvv,(1,0,2)).flatten()
    yvv=np.transpose(yvv,(1,0,2)).flatten()
    zvv=zvv.flatten()
    Points=np.vstack(([xvv,yvv,zvv])).transpose()
    Points/=nx
    return(Points)

from scipy.spatial import cKDTree

def Bresenham_3d(x0,y0,z0,x1,y1,z1):
    """Modified from https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/ """
    ListOfPoints = []
    ListOfPoints.append((x0, y0, z0))
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    xs = -1 if x0 > x1 else 1
    ys = -1 if y0 > y1 else 1
    zs = -1 if z0 > z1 else 1
  
    # Driving axis is X-axis"
    if (dx >= dy and dx >= dz):        
        p0 = 2 * dy - dx
        p1 = 2 * dz - dx
        while (x0 != x1):
            x0 += xs
            if (p0 >= 0):
                y0 += ys
                p0 -= 2 * dx
            if (p1 >= 0):
                z0 += zs
                p1 -= 2 * dx
            p0 += 2 * dy
            p1 += 2 * dz
            ListOfPoints.append((x0, y0, z0))
  
    # Driving axis is Y-axis"
    elif (dy >= dx and dy >= dz):       
        p0 = 2 * dx - dy
        p1 = 2 * dz - dy
        while (y0 != y1):
            y0 += ys
            if (p0 >= 0):
                x0 += xs
                p1 -= 2 * dy
            if (p1 >= 0):
                z0 += zs
                p0 -= 2 * dy
            p0 += 2 * dx
            p1 += 2 * dz
            ListOfPoints.append((x0, y0, z0))
  
    # Driving axis is Z-axis"
    else:        
        p0 = 2 * dy - dz
        p1 = 2 * dx - dz
        while (z0 != z1):
            z0 += zs
            if (p0 >= 0):
                y0 += ys
                p0 -= 2 * dz
            if (p1 >= 0):
                x0 += xs
                p1 -= 2 * dz
            p0 += 2 * dy
            p1 += 2 * dx
            ListOfPoints.append((x0, y0, z0))
    
    return(np.array(ListOfPoints).astype(int))

def find_lowest_point(x0, y0, z0, x1, y1, z1, Total_EDT):

    points = Bresenham_3d(x0, y0, z0, x1, y1, z1)
    
    func_total_edt = lambda point : Total_EDT[point[0],point[1],point[2]]
    
    edt_points = np.array(list(map(func_total_edt,points)))
    edt_diffs = edt_points[:-1]-edt_points[1:]

    return(points[np.sum(edt_diffs>0)])

def create_coords_nc(nx,ny,nz):
    XV = np.linspace(0,nx-1,nx).astype(int)
    YV = np.linspace(0,ny-1,ny).astype(int)
    ZV = np.linspace(0,nz-1,nz).astype(int)
    xvv, yvv, zvv = np.meshgrid(XV, YV, ZV)
    xvv=xvv.transpose().flatten()
    yvv=yvv.transpose().flatten()
    zvv=zvv.transpose().flatten()
    Points=np.transpose(np.stack(([xvv,yvv,zvv])))
    return(Points)

def find_better_points(points,Total_EDT,mask_1,plot=False): 
    
    
    nx,ny,nz = Total_EDT.shape
    borders = skimage.filters.laplace(mask_1, ksize=3, mask=None)
    mask_3 = ((mask_1+borders)>1).astype(bool)

    idx = points[:,0]*ny*nz + points[:,1]*nz + points[:,2]
    int_points = points[(1-mask_3).astype(bool).flatten()[idx]]#.astype(float)
    idx_int = int_points[:,0]*ny*nz + int_points[:,1]*nz + int_points[:,2]

    Points = create_coords_not_cube(Total_EDT.shape[0],Total_EDT.shape[1],Total_EDT.shape[2])*Total_EDT.shape[0]
    points_ext = Points[(mask_3).astype(bool).flatten()]#.astype(float)

    Tree = cKDTree(points_ext)
    dist,indices = Tree.query(int_points)
    dir_grad = (int_points - points_ext[indices])
    norm = np.linalg.norm(dir_grad,axis=1)
    dir_grad/=norm.reshape(-1,1)
    
    func_total_edt = lambda x: Total_EDT[x[0],x[1],x[2]]
    points_end = (int_points +dir_grad*(np.array(list(map(func_total_edt,int_points))).reshape(-1,1))).astype(int)

    points_end[:,0]=points_end[:,0].clip(0,Total_EDT.shape[0]-1)
    points_end[:,1]=points_end[:,1].clip(0,Total_EDT.shape[1]-1)
    points_end[:,2]=points_end[:,2].clip(0,Total_EDT.shape[2]-1)

    print("Score before : ",sum(list(map(func_total_edt,int_points))))
    New_points = []
    t1 = time()
    for i in range(len(int_points)): 
        
        New_points.append(find_lowest_point(int_points[i,0],int_points[i,1],int_points[i,2],points_end[i,0],points_end[i,1],points_end[i,2],Total_EDT))
        #if i%100 == 0 : 
        #    print("Iteration :",i,"Time elapsed :", time() - t1)
    New_points = np.array(New_points)
    print("Score after : ",sum(list(map(func_total_edt,New_points))))

    return(New_points)


def give_anchor_points(EDT,h,values_thresh,distance_thresh,min_dist=10,maxes = True):#, mins = True):

    nx,ny,nz = EDT.shape
    alpha = ny/nx
    beta = nz/nx
    
    Points = create_coords_not_cube(nx,ny,nz)
    Values = EDT.flatten()
    Anchor_Points = Points[Values<values_thresh]
    print(len(Anchor_Points))
    box_anchors = give_anchors_not_cube(1,alpha,beta)
    Points_delaunay = np.vstack([box_anchors,Anchor_Points])

    points = np.vstack((Points_delaunay))

    if maxes : 
        is_peak = peak_local_max(EDT, min_distance=min_dist,indices=False) # outputs bool image
        labels = label(is_peak)[0]
        merged_peaks = center_of_mass(is_peak, labels, range(1, np.max(labels)+1))
        merged_peaks = np.array(merged_peaks).round().astype(int)
        #print(merged_peaks)

        Anchors_bis=merged_peaks[:,0]*ny*nz+merged_peaks[:,1]*nz+merged_peaks[:,2]
        Points_local_max = Points[Anchors_bis]

    else : Points_local_max=None

    points=Adjust_distance_points_scipy_interp(points,h,threshold=distance_thresh)
    
    return(points,Points_local_max)


def double_filtering(Total_EDT,mask_1,threshold_on_EDT,distance_thresh,min_dist,h):
    
    points,Points_local_max = give_anchor_points(Total_EDT,h,values_thresh=threshold_on_EDT,distance_thresh=distance_thresh,min_dist=min_dist)
    
    points*=Total_EDT.shape[0]
    eps=1

    Indexes = (points[:,0]<eps).astype(int) + (points[:,1]<eps).astype(int) + (points[:,0]>Total_EDT.shape[0]-1-eps).astype(int) + (points[:,1]>Total_EDT.shape[1]-1-eps).astype(int)
    interior_points=points[Indexes==0].astype(int)
    exterior_points=points[Indexes>0]

    
    interior_points = find_better_points(interior_points,Total_EDT,mask_1,plot=False)
    interior_points=interior_points.astype(float)
    interior_points/=Total_EDT.shape[0]
    exterior_points/=Total_EDT.shape[0]
    interior_points = Adjust_distance_points_scipy_interp(interior_points,h,distance_thresh)
    
    nx,ny,nz = Total_EDT.shape
    borders=give_anchors_not_cube(nx,ny,nz)
    points = np.vstack((interior_points,exterior_points,Points_local_max,borders))#.astype(int)
    return(points)


def create_regularisation_shell(mask_1,size):
    m1 = mask_1.copy()
    for i in range(size): 
        borders = skimage.filters.laplace(m1, ksize=3, mask=None)
        shell = ((m1+borders)>1).astype(bool)
        m1 = (m1.astype(int)-shell.astype(int)).astype(bool)
    return(shell)

def create_regularisation_points(mask_1,size,h,EDT,distance_thresh):
    print("Size of the regularization shell",size)
    shell = create_regularisation_shell(mask_1,size)
    nx,ny,nz = EDT.shape
    
    Points = create_coords_not_cube(nx,ny,nz)
    points = Points[shell.flatten()]

    points=Adjust_distance_points_scipy_interp(points,h,threshold=distance_thresh)
    return(points)

def build_tesselation_and_EDT_from_membrane_prob(membrane_prob,val_thresh,point_density,size_shell='auto',plot_figure=True):
    
    DT = membrane_prob-np.amin(membrane_prob)
    DT/=np.amax(DT)
    DT=1-DT
    Total_EDT,threshold_on_EDT = auto_double_edt(DT,val_thresh,normalize=False)


    mask_1 = DT>val_thresh
    h = interpolate_map(Total_EDT)
    min_dist = np.ceil(threshold_on_EDT).astype(int)
    distance_thresh = threshold_on_EDT/(Total_EDT.shape[0]*point_density)
    
    print("Min_dist",min_dist)
    print("Distance_thresh",threshold_on_EDT/point_density)
    
    points = double_filtering(Total_EDT,mask_1,threshold_on_EDT,distance_thresh,min_dist,h)

    if size_shell =='auto' : 
        points_regularisation = create_regularisation_points(mask_1,min_dist*2,h,Total_EDT,distance_thresh)
    else : 
        points_regularisation = create_regularisation_points(mask_1,int(min_dist*size_shell),h,Total_EDT,distance_thresh)
    points = np.vstack((points,points_regularisation))

    tesselation=Delaunay(points)
    
    Total_EDT = gaussian(Total_EDT,sigma=1)  #BLUR IT OR NOT BLUR IT, THAT IS THE QUESTION
    Total_EDT-=np.amin(Total_EDT)
    Total_EDT/=np.amax(Total_EDT)
    
    return(tesselation,Total_EDT)


def plot_mask_to_evaluate_val_thresh(membrane_prob,val_thresh):
    
    DT = membrane_prob-np.amin(membrane_prob)
    DT/=np.amax(DT)
    DT=1-DT

    fig,axs=plt.subplots(1,2,figsize=(10,5))
    axs[0].imshow(DT[DT.shape[0]//2]>val_thresh)
    axs[1].imshow(DT[DT.shape[0]//2])

"""
def Bresenham_3d(x0,y0,z0,x1,y1,z1):
    #Modified from https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/ 
    ListOfPoints = []
    ListOfPoints.append((x0, y0, z0))
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    xs = -1 if x0 > x1 else 1
    ys = -1 if y0 > y1 else 1
    zs = -1 if z0 > z1 else 1
  
    # Driving axis is X-axis"
    if (dx >= dy and dx >= dz):        
        p0 = 2 * dy - dx
        p1 = 2 * dz - dx
        while (x0 != x1):
            x0 += xs
            if (p0 >= 0):
                y0 += ys
                p0 -= 2 * dx
            if (p1 >= 0):
                z0 += zs
                p1 -= 2 * dx
            p0 += 2 * dy
            p1 += 2 * dz
            ListOfPoints.append((x0, y0, z0))
  
    # Driving axis is Y-axis"
    elif (dy >= dx and dy >= dz):       
        p0 = 2 * dx - dy
        p1 = 2 * dz - dy
        while (y0 != y1):
            y0 += ys
            if (p0 >= 0):
                x0 += xs
                p1 -= 2 * dy
            if (p1 >= 0):
                z0 += zs
                p0 -= 2 * dy
            p0 += 2 * dx
            p1 += 2 * dz
            ListOfPoints.append((x0, y0, z0))
  
    # Driving axis is Z-axis"
    else:        
        p0 = 2 * dy - dz
        p1 = 2 * dx - dz
        while (z0 != z1):
            z0 += zs
            if (p0 >= 0):
                y0 += ys
                p0 -= 2 * dz
            if (p1 >= 0):
                x0 += xs
                p1 -= 2 * dz
            p0 += 2 * dy
            p1 += 2 * dx
            ListOfPoints.append((x0, y0, z0))
    
    return(ListOfPoints)

def find_lowest_point(x0, y0, z0, x1, y1, z1, Total_EDT):

    points = Bresenham_3d(x0, y0, z0, x1, y1, z1)
    
    func_total_edt = lambda point : Total_EDT[point[0],point[1],point[2]]
    edt_points = np.array(list(map(func_total_edt,points)))
    edt_diffs = edt_points[:-1]-edt_points[1:]

    return(points[np.sum(edt_diffs>0)])

def create_coords_nc(nx,ny,nz):
    XV = np.linspace(0,nx-1,nx).astype(int)
    YV = np.linspace(0,ny-1,ny).astype(int)
    ZV = np.linspace(0,nz-1,nz).astype(int)
    xvv, yvv, zvv = np.meshgrid(XV, YV, ZV)
    xvv=xvv.transpose().flatten()
    yvv=yvv.transpose().flatten()
    zvv=zvv.transpose().flatten()
    Points=np.transpose(np.stack(([xvv,yvv,zvv])))
    return(Points)

def find_better_points(points,Total_EDT,mask_1,plot=False): 
    
    Points = create_coords_nc(Total_EDT.shape[0],Total_EDT.shape[1],Total_EDT.shape[2])
    points_ext = Points[mask_1.flatten()]
    
    Tree = cKDTree(points_ext)
    dist,indices = Tree.query(points)
    dir_grad = (points - points_ext[indices]).astype(float)
    norm = np.linalg.norm(dir_grad,axis=1)
    dir_grad/=norm.reshape(-1,1)
     
    func_total_edt = lambda x: Total_EDT[x[0],x[1],x[2]]
    points_end = (points +dir_grad*(np.array(list(map(func_total_edt,points))).reshape(-1,1))).astype(int)
    
    points_end[:,0]=points_end[:,0].clip(0,Total_EDT.shape[0]-1)
    points_end[:,1]=points_end[:,1].clip(0,Total_EDT.shape[1]-1)
    points_end[:,2]=points_end[:,2].clip(0,Total_EDT.shape[2]-1)

    New_points = []
    for i in range(len(points)): 
        New_points.append(find_lowest_point(points[i,0],points[i,1],points[i,2],points_end[i,0],points_end[i,1],points_end[i,2],Total_EDT))
    New_points = np.array(New_points)

    return(New_points)


def give_anchor_points(DT,values_thresh,distance_thresh,min_dist=10):
    
    ##########
    ## Compute the anchor points used for the triangulation
    ## values_thresh defines the maximal value that a point can have to be defined as an anchor point
    ## distance_thresh is the minimal distance between two anchor points (in the box [0,1]*[0,alpha]) 
    ## min_dist is the minimal distance (in pixels) between two local minimas/maximas. 
    ## add_local_min_max is True if we add local minimas and maximas to the anchor points, otherwise they are not added
    ##########

    nx, ny, nz = DT.shape
    alpha=ny/nx
    beta=nz/nx

    Indices = np.arange(nx*ny*nz).reshape(DT.shape)
    Anchors = Indices[DT<values_thresh].flatten()

    Points = create_coords_nc(nx,ny,nz)

    f_DT = gaussian(DT,sigma=1)
    Points_local_max = peak_local_max(f_DT, min_distance=min_dist)

    
    h = interpolate_map(DT)
    
    points=Adjust_distance_points_scipy_interp(points,h,threshold=distance_thresh)


    return(points,Points_local_max)



def double_filtering(Total_EDT,mask_1,threshold_on_EDT,min_dist):
    distance_thresh = min_dist#/Total_EDT.shape[0]

    t1 = time
    points,Points_local_max = give_anchor_points(Total_EDT,values_thresh=threshold_on_EDT,distance_thresh=distance_thresh,min_dist=min_dist)
    print()
    eps=1
    Indexes = (points[:,0]<eps).astype(int) + (points[:,1]<eps).astype(int) + (points[:,2]<eps).astype(int) + (points[:,0]>Total_EDT.shape[0]-1-eps).astype(int) + (points[:,1]>Total_EDT.shape[1]-1-eps).astype(int) + (points[:,2]>Total_EDT.shape[2]-1-eps).astype(int)
    interior_points=points[Indexes==0].astype(int)
    exterior_points=points[Indexes>0]
    
    interior_points = find_better_points(interior_points,Total_EDT,mask_1,plot=False)

    h = interpolate_map(Total_EDT)
    
    interior_points = Adjust_distance_points_scipy_interp(interior_points,h,distance_thresh)
    
    points = np.vstack((interior_points,exterior_points,Points_local_max)).astype(int)
    idx_max_points = len(interior_points)
    return(points,idx_max_points)

    def build_tesselation_and_EDT_from_membrane_prob(membrane_prob,val_thresh,point_density,plot_figure=True):
    
    DT = membrane_prob-np.amin(membrane_prob)
    DT/=np.amax(DT)
    DT=1-DT
    Total_EDT,dist_points,threshold_on_EDT = auto_double_edt(DT,val_thresh,normalize=False)
    dist_points/=point_density
    min_dist = np.ceil(dist_points).astype(int)
    distance_thresh = min_dist

    print("Distance_threshold :",distance_thresh)

    mask_1 = DT>val_thresh
    #mask_1 = np.pad(mask_1, ((1, 1), (1, 1), (1,1)), 'constant',constant_values = 0)
    points,idx_max_interior_points = double_filtering(Total_EDT,mask_1,threshold_on_EDT,min_dist=min_dist)
    idx_interior_points = np.arange(idx_max_interior_points)

    points_f=points.astype(float)
    points_f/=Total_EDT.shape[0]

    tesselation=Delaunay(points_f)
    
    Total_EDT = gaussian(Total_EDT,sigma=1)  #BLUR IT OR NOT BLUR IT, THAT IS THE QUESTION
    Total_EDT-=np.amin(Total_EDT)
    Total_EDT/=np.amax(Total_EDT)
    
    return(tesselation,Total_EDT)
"""
def auto_double_edt(DT, val_thresh,normalize=True):
    
    mask_1 = DT>val_thresh
    mask_2 = (1-DT)>(1-val_thresh)
    mask_1 = np.pad(mask_1, ((1, 1), (1, 1), (1, 1)), 'constant',constant_values = 0)
    mask_2 = np.pad(mask_2, ((1, 1), (1, 1), (1, 1)), 'constant',constant_values = 1)
    EDT_1 = ndimage.distance_transform_edt(mask_1)
    EDT_2 = ndimage.distance_transform_edt(mask_2)
    inv = np.amax(EDT_2)-EDT_2
    Total_EDT = (EDT_1+np.amax(EDT_2))*mask_1 + inv*mask_2

    vt = np.amax(EDT_2)
    if normalize : 
        Total_EDT/=np.amax(Total_EDT)
        vt/=np.amax(Total_EDT)
    
    Total_EDT = Total_EDT[1:-1,1:-1,1:-1]
    return(Total_EDT,vt)


def Faces_score_from_sampling(Faces, Verts,f): 

    F = Faces
    V = Verts[F]

    Centroids = np.mean(V, axis = 1)
    Middle = (( V + V[:,[2,0,1]] )/2)
    OldMiddle = Middle.copy()
    V1 = V[:,0]
    V2 = V[:,1]
    V3 = V[:,2]
    
    M1 = Middle[:,0]
    M2 = Middle[:,1]
    M3 = Middle[:,2]

    P1 = (Centroids + V[:,0]) /2
    P2 = (Centroids + V[:,1]) /2
    P3 = (Centroids + V[:,2]) /2

    PM1 = (Centroids + OldMiddle[:,0]) /2
    PM2 = (Centroids+ OldMiddle[:,1]) /2
    PM3 = (Centroids + OldMiddle[:,2]) /2
    
    Score_Faces = np.zeros(len(Faces))
    
    Tuple = (Centroids,P1,P2,P3,PM1,PM2,PM3)#(
    #Tuple = (V1,V2,V3,M1,M2,M3,Centroids,P1,P2,P3,PM1,PM2,PM3)
    for arr in Tuple : 
        Score_Faces+=np.array(f(arr))
    
    return(Score_Faces/13)



def Faces_score_from_sampling_max(Faces,Verts,f):
    F = Faces
    V = Verts[F]

    Centroids = np.mean(V, axis = 1)
    Middle = (( V + V[:,[2,0,1]] )/2)
    OldMiddle = Middle.copy()

    V1 = V[:,0]
    V2 = V[:,1]
    V3 = V[:,2]
    
    M1 = Middle[:,0]
    M2 = Middle[:,1]
    M3 = Middle[:,2]

    P1 = (Centroids + V[:,0]) /2
    P2 = (Centroids + V[:,1]) /2
    P3 = (Centroids + V[:,2]) /2

    PM1 = (Centroids + OldMiddle[:,0]) /2
    PM2 = (Centroids+ OldMiddle[:,1]) /2
    PM3 = (Centroids + OldMiddle[:,2]) /2
    
    Tuple = (V1,V2,V3,M1,M2,M3,Centroids,P1,P2,P3,PM1,PM2,PM3)
    List_scores = []
    for elmt in Tuple : 
        List_scores.append(f(elmt))

    Values = np.vstack(List_scores).transpose()
    Scores = np.amax(Values,axis=1)

    return(Scores)
"""
def plot_total_EDT
def plot_DT 
DT = membrane_prob-np.amin(membrane_prob)
    DT/=np.amax(DT)
    DT=1-DT
    Total_EDT,dist_points,threshold_on_EDT = auto_double_edt(DT,val_thresh,normalize=False)"""