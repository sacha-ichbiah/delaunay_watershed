from Networkx_functions import seeded_watershed_map
from Plotting_tools import compute_seeds_idx_from_pixel_coords, plot_polylines_ax,retrieve_border_lines,plot_triangles_coloured_ax,plot_triangulation_ax
from Graph_functions_2D import Delaunay_Graph
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from skimage.segmentation import expand_labels
from Geometric_utilities_2D import build_triangulation
from Dcel import DCEL_Data, Clean_mesh
#from Dcel import DCEL_Data,Clean_mesh_from_seg


class geometry_reconstruction_2d():

    def __init__(self, labels,min_dist = 4 ,interpolation_points=10,expansion_labels = 0,original_image = None):
        
        self.labels= expand_labels(labels,expansion_labels)
        self.seeds_coords,self.seeds_indices, self.tri, self.EDT = build_triangulation(self.labels,min_distance=min_dist)
        self.original_image = original_image


        x = np.linspace(0,self.EDT.shape[0]-1,self.EDT.shape[0])  #To determine : start from 1 and end at EDT.shape[] or from  and end at EDT.shape[-1] ?
        y = np.linspace(0,self.EDT.shape[1]-1,self.EDT.shape[1])
        edt = interpolate.interp2d(x, y, self.EDT.transpose(), kind='linear')
        label = interpolate.interp2d(x, y, self.labels.transpose(), kind='linear')
        
        self.Delaunay_Graph = Delaunay_Graph(self.tri, edt,label,self.EDT.shape[0],self.EDT.shape[1],npoints=interpolation_points)
        self.build_graph()
        self.watershed_seeded()

    def build_graph(self):
        self.Nx_Graph = self.Delaunay_Graph.networkx_graph_weights_and_borders()


    def watershed_seeded(self, print_info=False,plot_figure=False):
        seeds_nodes = compute_seeds_idx_from_pixel_coords(self.EDT,self.Delaunay_Graph.compute_nodes_centroids(),self.seeds_coords,plot_figure = plot_figure)
        zero_nodes = self.Delaunay_Graph.compute_zero_nodes()
        self.Map_end = seeded_watershed_map(self.Nx_Graph,seeds_nodes,self.seeds_indices,zero_nodes)


    def return_mesh(self):
        return(Clean_mesh(self))
    
    def return_dcel(self): 
        v,e = self.return_mesh()
        return(DCEL_Data(v,e))

    def simple_plot(self): 
        fig, axs = plt.subplots(1,3,figsize = (24,8))
        axs[0].imshow(self.labels,cmap = plt.cm.nipy_spectral)
        axs[0].set_title("Labels")
        
        axs[1].imshow(self.EDT,plt.cm.magma)
        axs[1].set_title("Polygonal lines")
        Lines = retrieve_border_lines(self.Delaunay_Graph, self.Map_end)
        plot_polylines_ax(axs[1],Lines,self.Delaunay_Graph.Vertices)
        
        plot_triangles_coloured_ax(axs[2],self.Delaunay_Graph, self.Map_end)
        axs[2].set_title("Triangles")

        for ax in axs : 
            ax.set_xticks([])
            ax.set_yticks([])
        plt.subplots_adjust(wspace=0.05, hspace=0)
        
        
    def extended_plot(self): 
        
        fig, axs = plt.subplots(1,6,figsize = (32,12))
        axs[0].imshow(self.original_image,plt.cm.gray)
        axs[0].set_title("Original Image")
        axs[1].imshow(self.labels,cmap = plt.cm.nipy_spectral)
        axs[1].set_title("Labels")
        
        axs[2].imshow(self.EDT,plt.cm.gray)
        axs[2].scatter(self.seeds_coords[:,1],self.seeds_coords[:,0],c = np.random.rand(len(self.seeds_coords),3),s=180)
        axs[2].set_title("Seeds from EDT")
        
        axs[3].imshow(self.EDT,plt.cm.gray)
        axs[3].set_title("Delaunay triangulation")
        plot_triangulation_ax(axs[3],self)
        
        
        axs[4].imshow(self.EDT,plt.cm.magma)
        axs[4].set_title("Polygonal Lines")
        Lines = retrieve_border_lines(self.Delaunay_Graph, self.Map_end)
        plot_polylines_ax(axs[4],Lines,self.Delaunay_Graph.Vertices)
        
        axs[5].set_title("Triangles")
        plot_triangles_coloured_ax(axs[5],self.Delaunay_Graph, self.Map_end)
        
        for ax in axs : 
            ax.set_xticks([])
            ax.set_yticks([])
        plt.subplots_adjust(wspace=0.0, hspace=0)



