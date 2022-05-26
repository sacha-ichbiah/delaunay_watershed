from scipy import ndimage
from transforms import StandardLabelToBoundary, LabelToAffinities, LabelToZAffinities
from skimage.feature import peak_local_max
import numpy as np
from scipy.spatial import Delaunay

def pad_mask(mask,pad_size = 1): 
    padded_mask = mask.copy()[pad_size:-pad_size,pad_size:-pad_size]
    padded_mask = np.pad(padded_mask, ((pad_size, pad_size), (pad_size, pad_size)), 'constant',constant_values = 1)
    return(padded_mask)
def give_corners(img): 
    Points=np.zeros((4,2))
    index=0
    a,b = img.shape
    for i in [0,a-1]: 
        for j in [0,b-1]: 
            Points[index]=np.array([i,j])
            index+=1
    return(Points)

def build_triangulation(labels,min_distance=4):
    b = StandardLabelToBoundary()(labels.reshape(1,labels.shape[0],labels.shape[1]))[0,0]
    mask_2 = b
    EDT_2 = ndimage.distance_transform_edt(mask_2)
    b = pad_mask(b)
    mask_1 = 1-b
    EDT_1 = ndimage.distance_transform_edt(mask_1)
    inv = np.amax(EDT_2)-EDT_2
    Total_EDT = (EDT_1+np.amax(EDT_2))*mask_1 + inv*mask_2

    seeds_coords = []

    values_lbls = np.unique(labels) 
    for i in values_lbls:
        seed = np.argmax(Total_EDT*((labels==i).astype(float)))
        seeds_coords.append([seed//labels.shape[1],seed%labels.shape[1]])
        #break
    seeds_coords = np.array(seeds_coords)

    points = peak_local_max(-Total_EDT,min_distance=min_distance,exclude_border=False)
    local_maxes = peak_local_max(Total_EDT,min_distance=min_distance,exclude_border=False)
    corners = give_corners(Total_EDT)

    all_points = np.vstack((points,corners,local_maxes))
    tesselation=Delaunay(all_points)
    
    return(seeds_coords, tesselation,Total_EDT)

