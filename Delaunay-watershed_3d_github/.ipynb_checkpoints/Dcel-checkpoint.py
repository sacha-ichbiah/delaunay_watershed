#Sacha Ichbiah, Sept 2021
"""Module dedicated to the representation of a DCEL graph."""
# Define the structures of a double connected edge list (https://en.wikipedia.org/wiki/Doubly_connected_edge_list)
from dataclasses import dataclass, field
import math
import pickle
from typing import List
import numpy as np 
from Mesh_utilities import separate_faces_dict
from Curvature import compute_curvature_interfaces
from Mesh_Geometry import compute_areas_faces,compute_areas_cells,compute_angles_tri,compute_angles_tri_and_quad,compute_volume_cells,compute_areas_interfaces,compute_volume_derivative_matrix,compute_area_derivative_dict

def find_key_multiplier(num_points): 
    key_multiplier = 1
    while num_points//key_multiplier != 0 : 
        key_multiplier*=10
    return(key_multiplier)   

@dataclass
class Vertex:
    """Vertex in 2D"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    key: int=0
    on_trijunction = False

 

@dataclass
class HalfEdge:
    """Half-Edge of a DCEL graph"""

    origin: Vertex = None
    destination: Vertex = None
    material_1: int = 0
    material_2: int = 0
    twin = None
    incident_face: "Face" = None
    prev: "HalfEdge" = None
    next: "HalfEdge" = None
    attached: dict = field(default_factory=dict)
    key: int=0

    def compute_length(self): 
        v = np.zeros(3)
        v[0]=self.origin.x-self.destination.x
        v[1]=self.origin.y-self.destination.y
        v[2]=self.origin.z-self.destination.z
        self.length = np.linalg.norm(v)

    def set_face(self, face):
        if self.incident_face is not None:
            print("Error : the half-edge already has a face.")
            return
        self.incident_face = face
        if self.incident_face.outer_component is None:
            face.outer_component = self

    def set_prev(self, other):
        if other.incident_face is not self.incident_face:
            print("Error setting prev relation : edges must share the same face.")
            return
        self.prev = other
        other.next = self

    def set_next(self, other):
        if other.incident_face is not self.incident_face:
            print("Error setting next relation : edges must share the same face.")
            return
        self.next = other
        other.prev = self

    def set_twin(self, other):
        self.twin = other
        other.twin = other

    def return_vector(self): 
        xo,yo = self.origin.x,self.origin.y
        xt,yt = self.destination.x,self.destination.y
        vect = np.array([xt-xo,yt-yo])
        vect/=np.linalg.norm(vect)
        return(vect)

    def __repr__(self):
        ox = "None"
        oy = "None"
        dx = "None"
        dy = "None"
        if self.origin is not None:
            ox = str(self.origin.x)
            oy = str(self.origin.y)
        if self.destination is not None:
            dx = str(self.destination.x)
            dy = str(self.destination.y)
        return f"origin : ({ox}, {oy}) ; destination : ({dx}, {dy})"


@dataclass
class Face:
    """Face of a DCEL graph"""

    attached: dict = field(default_factory=dict)
    outer_component: HalfEdge = None
    _closed: bool = True
    material_1: int = 0
    material_2: int = 0
    normal = None
    key: int=0

    # def set_outer_component(self, half_edge):
    #     if half_edge.incident_face is not self:
    #         print("Error : the edge must have the same incident face.")
    #         return
    #     self.outer_component = half_edge

    def first_half_edge(self):
        self._closed = False
        first_half_edge = self.outer_component
        if first_half_edge is None:
            return None
        while first_half_edge.prev is not None:
            first_half_edge = first_half_edge.prev
            if first_half_edge is self.outer_component:
                self._closed = True
                break
        return first_half_edge

    def last_half_edge(self):
        self._closed = False
        last_half_edge = self.outer_component
        if last_half_edge is None:
            return None
        while last_half_edge.next is not None:
            last_half_edge = last_half_edge.next
            if last_half_edge is self.outer_component:
                self._closed = True
                last_half_edge = self.outer_component.prev
                break
        return last_half_edge

    def closed(self):
        self.first_half_edge()
        return self._closed

    def get_edges(self):
        edges = []
        if self.outer_component is None:
            return edges

        first_half_edge = self.first_half_edge()
        last_half_edge = self.last_half_edge()
        edge = first_half_edge
        while True:
            edges.append(edge)
            if edge is last_half_edge:
                break
            else:
                edge = edge.next
        return edges

    def get_materials(self): 
        self.material_1 = self.outer_component.material_1
        self.material_2 = self.outer_component.material_2

    def get_vertices(self):
        vertices = []
        for edge in self.get_edges():
            if edge.origin is not None:
                vertices.append(edge.origin)
        return vertices

    def get_area(self):
        if not self.closed():
            return None
        else:

            def distance(p1, p2):
                return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + ((p1.z - p2.z) ** 2))

            area = 0
            vertices = self.get_vertices()
            p1 = vertices[0]
            for i in range(1, len(vertices) - 1):
                p2 = vertices[i]
                p3 = vertices[i + 1]
                a = distance(p1, p2)
                b = distance(p2, p3)
                c = distance(p3, p1)
                s = (a + b + c) / 2.0
                area += math.sqrt(s * (s - a) * (s - b) * (s - c))
            return area


class DCEL_Data:
    """DCEL Graph containing faces, half-edges and vertices."""
    #Take a multimaterial mesh as an entry
    def __init__(self,Verts,Faces):
        for i, f in enumerate(Faces): 
            if f[3]>f[4]: 
                Faces[i]=Faces[i,[0,2,1,4,3]]
        
        Verts,Faces = remove_unused_vertices(Verts,Faces)
        for i in range(len(Faces)): 
            Faces[i]=Faces[i,[0,2,1,3,4]]
        self.v = Verts
        self.f = Faces
        self.materials = np.unique(Faces[:,[3,4]])
        self.n_materials = len(self.materials)
        Vertices_list, Halfedges_list, Faces_list = build_lists(Verts,Faces)
        self.vertices = Vertices_list
        self.faces = Faces_list
        self.half_edges = Halfedges_list
        self.compute_areas_faces()
        self.compute_centroids_cells()
        self.mark_trijunctional_vertices()
        self.compute_length_halfedges()

    def compute_length_halfedges(self): 
        compute_length_halfedges(self)

    def compute_areas_faces(self):
        compute_areas_faces(self)
    
    def compute_angles_tri(self,unique=True):
        return(compute_angles_tri(self,unique=unique))

    def compute_angles_tri_and_quad(self,unique=True):
        return(compute_angles_tri_and_quad(self,unique=unique))

    def compute_curvatures_interfaces(self,laplacian="robust",weighted=True):
        #"robust" or "cotan"
        return(compute_curvature_interfaces(self,laplacian=laplacian,weighted=weighted))

    def compute_centroids_cells(self): 
        self.centroids = {}
        separated_faces = separate_faces_dict(self.f)
        for i in separated_faces.keys(): 
            self.centroids[i]=np.mean(self.v[np.unique(separated_faces[i])],axis=0)
        
    def mark_trijunctional_vertices(self,return_list=False): 
        return(mark_trijunctional_vertices(self,return_list))
        

    def compute_areas_cells(self):
        return(compute_areas_cells(self))

    def compute_areas_interfaces(self): 
        return(compute_areas_interfaces(self))
    
    def compute_area_derivatives(self): 
        return(compute_area_derivative_dict(self))

    def compute_volumes_cells(self): 
        return(compute_volume_cells(self))
    
    def compute_volume_derivatives(self):
        return(compute_volume_derivative_matrix(self))

    def compute_vertex_normals(self): 
        return(compute_vertex_normals(self.v,self.f))
    
    def find_trijunctional_edges(self):
        return(find_trijunctional_edges(self))


    def save(self, filename):
        with open(filename, "wb") as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.vertices, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.half_edges, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.faces, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, "rb") as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            self.vertices = pickle.load(f)
            self.half_edges = pickle.load(f)
            self.faces = pickle.load(f)
            
class DCEL_Data_from_seg:
    """DCEL Graph containing faces, half-edges and vertices."""
    #Take a multimaterial mesh as an entry
    def __init__(self,Seg):
        
                
        V, Faces, Faces_idx, Nodes_linked = retrieve_mesh_multimaterial_multitracker_format(Seg.Delaunay_Graph,Seg.Map_end)
        Verts = V.copy()
        #shapes = Seg.image.shape
        #Verts[:,0]*=shapes[1]/shapes[0]
        
        for i, f in enumerate(Faces): 
            if f[3]>f[4]: 
                Faces[i]=Faces[i,[0,1,2,4,3]]
                Nodes_linked[i]=Nodes_linked[i][[1,0]]
                
       
        Faces = reorient_faces(Faces,Seg,Nodes_linked)
        #Verts,Faces = remove_unused_vertices(Verts,Faces)
        for i in range(len(Faces)): 
            Faces[i]=Faces[i,[0,2,1,3,4]]
                
        self.v = Verts
        self.f = Faces
        self.materials = np.unique(Faces[:,[3,4]])
        self.n_materials = len(self.materials)
        Vertices_list, Halfedges_list, Faces_list = build_lists(Verts,Faces)
        self.vertices = Vertices_list
        self.faces = Faces_list
        self.half_edges = Halfedges_list
        self.compute_areas_faces()
        self.compute_centroids_cells()
        self.mark_trijunctional_vertices()
        self.compute_length_halfedges()
    
    def compute_length_halfedges(self): 
        compute_length_halfedges(self)

    def compute_areas_faces(self):
        compute_areas_faces(self)
    
    def compute_angles_tri(self,unique=True):
        return(compute_angles_tri(self,unique=unique))

    def compute_angles_tri_and_quad(self,unique=True):
        return(compute_angles_tri_and_quad(self,unique=unique))

    def compute_curvatures_interfaces(self,laplacian="robust",weighted=True):
        #"robust" or "cotan"
        return(compute_curvature_interfaces(self,laplacian=laplacian,weighted=weighted))

    def compute_centroids_cells(self): 
        self.centroids = {}
        separated_faces = separate_faces_dict(self.f)
        
        for i in separated_faces.keys():
            self.centroids[i]=np.mean(self.v[np.unique(separated_faces[i]).astype(int)],axis=0)
        
    def mark_trijunctional_vertices(self,return_list=False): 
        return(mark_trijunctional_vertices(self,return_list))
        

    def compute_areas_cells(self):
        return(compute_areas_cells(self))

    def compute_areas_interfaces(self): 
        return(compute_areas_interfaces(self))
    
    def compute_area_derivatives(self): 
        return(compute_area_derivative_dict(self))

    def compute_volumes_cells(self): 
        return(compute_volume_cells(self))
    
    def compute_volume_derivatives(self):
        return(compute_volume_derivative_matrix(self))

    def compute_vertex_normals(self): 
        return(compute_vertex_normals(self.v,self.f))

    def find_trijunctional_edges(self):
        return(find_trijunctional_edges(self))

    def save(self, filename):
        with open(filename, "wb") as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.vertices, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.half_edges, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.faces, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, "rb") as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            self.vertices = pickle.load(f)
            self.half_edges = pickle.load(f)
            self.faces = pickle.load(f)



def retrieve_mesh_multimaterial_multitracker_format(Graph,Map):
    ##Must be used without any filtering operation
    reverse_map ={}
    for key in Map : 
        for node_idx in Map[key] :
            reverse_map[node_idx]=key
    Faces=[]
    Faces_idx = []
    Nodes_linked = []
    for idx in range(len(Graph.Faces)) : 
        nodes_linked = Graph.Nodes_Linked_by_Faces[idx]
        

        cluster_1 = reverse_map[nodes_linked[0]]#reverse_map.get(nodes_linked[0],-1)
        cluster_2 = reverse_map[nodes_linked[1]]#reverse_map.get(nodes_linked[1],-2)
        #if the two nodes of the edges belong to the same cluster we ignore them
        #otherwise we add them to the mesh
        if cluster_1 != cluster_2 : 
            
            #If one thing has been filtered we add it to the background
            #If both have been filtered the interface is considered to not exist
            #if cluster_1 == -1 and cluster_2 ==-2 :
            #    continue
            #if cluster_2 == -2 : 
            #    cluster_2 = max(Map.keys())+1 
            #if cluster_1 ==-1 : 
            #    cluster_1 = max(Map.keys())+1 
                
            face = Graph.Faces[idx]
            cells = [cluster_1,cluster_2]
            Faces.append([face[0],face[1],face[2],cells[0],cells[1]])
            Faces_idx.append(idx)
            Nodes_linked.append(nodes_linked)

    for idx in range(len(Graph.Lone_Faces)):
        face = Graph.Lone_Faces[idx]
        node_linked = Graph.Nodes_linked_by_lone_faces[idx]
        cluster_1 = reverse_map[node_linked]
        #We incorporate all these edges because they are border edges
        if cluster_1 !=0:
            cells = [0,cluster_1]
            Faces.append([face[0],face[1],face[2],cells[0],cells[1]])
            Faces_idx.append(idx)
            Nodes_linked.append(nodes_linked)

    return(Graph.Vertices, np.array(Faces),Faces_idx,np.array(Nodes_linked))

def reorient_faces(Faces,Seg,Nodes_linked):
       
    #Thumb rule for all the faces
    
    Normals = compute_normal_Faces(Seg.Delaunay_Graph.Vertices,Faces)
    
    P = Seg.Delaunay_Graph.Vertices[Faces[:,:3]]
    Centroids_faces = np.mean(P,axis=1)
    Centroids_nodes = np.mean(Seg.Delaunay_Graph.Vertices[Seg.Delaunay_Graph.Tetra[Nodes_linked[:,0]]],axis=1)
    Vectors = Centroids_nodes-Centroids_faces
    Norms = np.linalg.norm(Vectors,axis=1)
    Vectors[:,0]/=Norms
    Vectors[:,1]/=Norms
    Vectors[:,2]/=Norms

    Dot_product = np.sum(np.multiply(Vectors,Normals),axis=1)
    Normals_sign = np.sign(Dot_product)
    
    #Reorientation according to the normal sign
    reoriented_faces = Faces.copy()
    for i,s in enumerate(Normals_sign) : 
        if s <0 : 
            reoriented_faces[i]=reoriented_faces[i][[0,2,1,3,4]]
            
    return(reoriented_faces)

"""
DCEL BUILDING FUNCTIONS
"""
def compute_normal_Faces(Verts,Faces):
    Pos = Verts[Faces[:,[0,1,2]]]
    Sides_1 = Pos[:,1]-Pos[:,0]
    Sides_2 = Pos[:,2]-Pos[:,1]
    Normal_faces = np.cross(Sides_1,Sides_2,axis=1)
    Norms = np.linalg.norm(Normal_faces,axis=1)#*(1+1e-8)
    Normal_faces/=(np.array([Norms]*3).transpose())
    return(Normal_faces)

def build_lists(Verts, Faces):
    Normals = compute_normal_Faces(Verts,Faces)
    Vertices_list = make_vertices_list(Verts)
    Halfedge_list = []
    for i in range(len(Faces)): 
        a,b,c,_,_ = Faces[i]
        Halfedge_list.append(HalfEdge(origin = Vertices_list[a], destination= Vertices_list[b],key=3*i+0))
        Halfedge_list.append(HalfEdge(origin = Vertices_list[c], destination= Vertices_list[a],key=3*i+1))
        Halfedge_list.append(HalfEdge(origin = Vertices_list[b], destination= Vertices_list[c],key=3*i+2))
        
    index=0
    for i in range(len(Faces)): 
        Halfedge_list[index].next = Halfedge_list[index+1]
        Halfedge_list[index].prev = Halfedge_list[index+2]
        
        Halfedge_list[index+1].next = Halfedge_list[index+2]
        Halfedge_list[index+1].prev = Halfedge_list[index]
        
        Halfedge_list[index+2].next = Halfedge_list[index]
        Halfedge_list[index+2].prev = Halfedge_list[index+1]
        
        index+=3
        
    Faces_list = []
    for i in range(len(Faces)): 
        Faces_list.append(Face(outer_component=Halfedge_list[i+3] ,material_1 = Faces[i,3], material_2 = Faces[i,4],key=i))
        Faces_list[i].normal = Normals[i]

    for i in range(len(Faces)): 
        Halfedge_list[3*i+0].incident_face = Faces_list[i]
        Halfedge_list[3*i+1].incident_face = Faces_list[i]
        Halfedge_list[3*i+2].incident_face = Faces_list[i]
        
    #find twins
    F = Faces.copy()[:,[0,1,2]]
    E = np.hstack((F,F)).reshape(-1,2)
    E = np.sort(E,axis=1)
    key_mult = find_key_multiplier(np.amax(F))
    Keys = E[:,1]*key_mult + E[:,0]
    Dict_twins = {}
    for i,key in enumerate(Keys) : 
        Dict_twins[key] = Dict_twins.get(key,[])+[i]
    List_twins = []
    counts = np.zeros(4)
    for i in range(len(E)):
        key = Keys[i]
        l = Dict_twins[key].copy()
        l.remove(i)
        List_twins.append(l)
        
    for i,list_twin in enumerate(List_twins):
        Halfedge_list[i].twin = list_twin

    return(Vertices_list,Halfedge_list,Faces_list)

def make_vertices_list(Verts): 
    Vertices_list = []
    for i,vertex_coords in enumerate(Verts) : 
        x,y,z = vertex_coords
        Vertices_list.append(Vertex(x=x,y=y,z=z,key=i))
    return(Vertices_list)


def mark_trijunctional_vertices(Mesh,return_list = False): 
    list_trijunctional_vertices = []
    for edge in Mesh.half_edges : 
        if len(edge.twin)>1 : 
            Mesh.vertices[edge.origin.key].on_trijunction = True
            Mesh.vertices[edge.destination.key].on_trijunction = True
            list_trijunctional_vertices.append(edge.origin.key)
            list_trijunctional_vertices.append(edge.destination.key)
    if return_list : 
        return(np.unique(list_trijunctional_vertices))


"""
DCEL Geometry functions
"""

def find_trijunctional_edges(Mesh):
    F = Mesh.f
    E = np.vstack((F[:,[0,1]],F[:,[0,2]],F[:,[1,2]]))
    key_mult = find_key_multiplier(len(Mesh.v)+1)
    K = (E[:,0]+1) + (E[:,1]+1)*key_mult
    Array,Index_first_occurence,Index_inverse,Index_counts = np.unique(K, return_index=True, return_inverse=True, return_counts=True)
    print("Number of trijunctional edges :",np.sum(Index_counts==3))
    Edges_trijunctions = E[Index_first_occurence]
    #Verts_concerned = np.unique(Edges_trijunctions)
    return(Edges_trijunctions)

def compute_length_halfedges(Mesh):
    for edge in Mesh.half_edges :
        edge.compute_length()

def compute_faces_areas(Verts,Faces):
    Pos = Verts[Faces[:,[0,1,2]]]
    Sides = Pos-Pos[:,[2,0,1]]
    Lengths_sides = np.linalg.norm(Sides,axis = 2)
    Half_perimeters = np.sum(Lengths_sides,axis=1)/2

    Diffs = np.array([Half_perimeters]*3).transpose() - Lengths_sides
    Areas = (Half_perimeters*Diffs[:,0]*Diffs[:,1]*Diffs[:,2])**(0.5)
    return(Areas)

def compute_vertex_normals(Verts,Faces): 
    faces_on_verts = [[] for x in range(len(Verts))]
    for i,f in enumerate(Faces) : 
        faces_on_verts[f[0]].append(i)
        faces_on_verts[f[1]].append(i)
        faces_on_verts[f[2]].append(i)

    Sides = Verts[Faces[:,[0,1,2]]]
    Side_1 = Sides[:,0]-Sides[:,1]
    Side_2 = Sides[:,0]-Sides[:,2]
    Faces_normals = np.cross(Side_1,Side_2,axis=1)
    norms = np.linalg.norm(Faces_normals, axis=1)
    Faces_normals*=np.array([1/norms]*3).transpose()
    Faces_areas = compute_faces_areas(Verts,Faces)
    vertex_normals = np.zeros(Verts.shape)

    for i,f_list in enumerate(faces_on_verts) : 
        c=0
        n=0
        for f_idx in f_list : 
            n+=Faces_normals[f_idx]*Faces_areas[f_idx]
            c+=Faces_areas[f_idx]
        n/=c
        vertex_normals[i]=n
    return(vertex_normals)


def remove_unused_vertices(V,F):
    #Some unused vertices appears after the tetrahedral remeshing. We need to remove them. 
    Verts = V.copy()
    Faces = F.copy()
    faces_on_verts = [[] for x in range(len(Verts))]
    for i,f in enumerate(Faces) : 
        faces_on_verts[f[0]].append(i)
        faces_on_verts[f[1]].append(i)
        faces_on_verts[f[2]].append(i)

    verts_to_remove = []
    for i,f_list in enumerate((faces_on_verts)):
        if len(f_list)==0 : 
            verts_to_remove.append(i)
    
    #print(len(verts_to_remove))
    
    list_verts = np.delete(np.arange(len(Verts)),verts_to_remove)
    idx_new_verts = np.arange(len(list_verts))
    mapping = dict(zip(list_verts,idx_new_verts))

    Verts = Verts[list_verts]
    for i in range(len(Faces)): 
        Faces[i,0]=mapping[Faces[i,0]]
        Faces[i,1]=mapping[Faces[i,1]]
        Faces[i,2]=mapping[Faces[i,2]]
    return(Verts,Faces)
