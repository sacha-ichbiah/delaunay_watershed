from Networkx_functions import seeded_watershed_map
from Graph_functions_3D import Delaunay_Graph
from Geometric_utilities_3D import build_triangulation, interpolate_image
from Mesh_utilities import write_mesh_text,write_mesh_bin, Clean_mesh_from_seg, plot_cells_polyscope,compute_seeds_idx_from_voxel_coords,retrieve_border_tetra_with_index_map
from skimage.segmentation import expand_labels
from time import time
import numpy as np 

class geometry_reconstruction_3d():
    def __init__(self,labels,min_dist = 4, expansion_labels = 0,original_image = None,print_info = False,mode='torch'):
        self.original_image = original_image
        if expansion_labels>0:
            self.labels= expand_labels(labels,expansion_labels)
        else : 
            self.labels = labels
        
        self.seeds_coords,self.seeds_indices, self.tri, self.EDT = build_triangulation(self.labels,min_distance=min_dist,prints=print_info,mode=mode)
        
        labels = interpolate_image(self.labels)
        edt = interpolate_image(self.EDT)
        self.Delaunay_Graph = Delaunay_Graph(self.tri, edt, labels,print_info = print_info)
        self.build_graph()        
        self.watershed_seeded(print_info=print_info)

    def build_graph(self):
        self.Nx_Graph = self.Delaunay_Graph.networkx_graph_weights_and_borders()

    def watershed_seeded(self,print_info=True,plot_figure=False):
        t1 = time()
        seeds_nodes = compute_seeds_idx_from_voxel_coords(self.EDT,self.Delaunay_Graph.compute_nodes_centroids(),self.seeds_coords)
        zero_nodes = self.Delaunay_Graph.compute_zero_nodes()
        self.Map_end = seeded_watershed_map(self.Nx_Graph,seeds_nodes,self.seeds_indices,zero_nodes)#Seeded_Watershed(self.Nx_Graph,seeds_nodes,self.seeds_indices,zero_nodes)
        
        ##if print_info : 
        #   print("Number of Nodes :",len(self.Nx_Graph.nodes),"Number of Edges :",len(self.Nx_Graph.edges))
        t2 = time()
        if print_info : print("Watershed done in ",np.round(t2-t1,3))

    def retrieve_clusters(self): 
        Clusters=retrieve_border_tetra_with_index_map(self.Delaunay_Graph,self.Map_end)
        return(Clusters)

    def return_mesh(self): 
        return(Clean_mesh_from_seg(self))

    def plot_cells_polyscope(self,anisotropy_factor = 1.): 
        Verts,Faces = self.return_mesh()
        Verts[:,0]*=anisotropy_factor
        plot_cells_polyscope(Verts,Faces)

    def export_mesh(self,filename,mode='bin'):
        Verts,Faces = self.return_mesh()
        if mode=='bin':
            write_mesh_text(filename, Verts, Faces)
        elif mode=='txt': 
            write_mesh_bin(filename, Verts, Faces)
        else : 
            print("Please choose a valid format")

    def plot_in_napari(self):
        import napari
        v = napari.view_image(self.labels,name='Labels')
        v.add_image(self.EDT,name='Distance Transform')
        if self.original_image is not None :
            v.add_image(self.original_image,name="Original Image")
        v.add_points(self.seeds_coords, name='Watershed seeds',n_dimensional=True,face_color = np.random.rand(len(self.seeds_coords),3),size = 10)
        v.add_points(self.tri.points, name='triangulation_points', n_dimensional=False, face_color = 'red', size = 1)
        return(v)

      

