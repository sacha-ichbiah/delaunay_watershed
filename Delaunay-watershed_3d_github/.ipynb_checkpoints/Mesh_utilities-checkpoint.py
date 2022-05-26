import numpy as np 
import struct
import polyscope as ps
from matplotlib import cm


def separate_faces_dict(Faces,n_towers=10): 
    n_towers = np.amax(Faces[:,[3,4]])+1
   
    Occupancy=np.zeros(n_towers)
    Dict={}
    for face in Faces : 
        _,_,_,num1,num2 = face
        if num1!=-1:
            if Occupancy[num1]==0:
                Dict[num1]=[face[[0,1,2]]]
                Occupancy[num1]+=1
            else : 
                Dict[num1].append(face[[0,1,2]])
            
        if num2!=-1:
            if Occupancy[num2]==0:
                Dict[num2]=[face[[0,1,2]]]
                Occupancy[num2]+=1
            else : 
                Dict[num2].append(face[[0,1,2]])
            
            
    Faces_separated={}
    for i in sorted(Dict.keys()) : 
        Faces_separated[i] = (np.array(Dict[i]))
        
    return(Faces_separated)

def write_mesh_bin(filename, Faces, Verts):
    assert(len(Faces[0])==5 and len(Verts[0])==3)
    strfile = struct.pack("Q", len(Verts))
    strfile +=Verts.flatten().astype(np.float64).tobytes()
    strfile += struct.pack("Q", len(Faces))
    dt=np.dtype([('triangles',np.uint64,(3,)), ('labels',np.int32,(2,))])
    F_n = Faces[:,:3].astype(np.uint64)
    F_t = Faces[:,3:].astype(np.int32)
    func = lambda i : (F_n[i],F_t[i])
    T=np.array(list(map(func,np.arange(len(Faces)))),dtype=dt)
    strfile+=T.tobytes()
    file = open(filename,'wb')
    file.write(strfile)
    file.close()


def write_mesh_text(filename, Verts, Faces):
    
    file = open(filename, 'w')
    file.write(str(len(Verts))+'\n')
    for i in range(len(Verts)): 
        file.write(f'{Verts[i][0]:.5f} {Verts[i][1]:.5f} {Verts[i][2]:.5f}'+'\n')
    file.write(str(len(Faces))+'\n')
    for i in range(len(Faces)): 
        file.write(f'{Faces[i][0]} {Faces[i][1]} {Faces[i][2]} {Faces[i][3]} {Faces[i][4]}'+'\n')
    file.close() 

def open_mesh_multitracker(type_file,filename="../../../../../Dropbox/S.Ichbiah/Data/Doublet images/cells2_alpha0.25_beta1.0_delta1.0-1.6/mesh000000.rec"): 
    if type_file == 'bin' : 
        return(read_rec_file_bin(filename))
    else : 
        return(read_rec_file_num(filename))

def read_rec_file_bin(filename): 
    mesh_file=open(filename,'rb')
    
    #Vertices
    num_vertices,=struct.unpack('Q', mesh_file.read(8)) 
    Verts=np.fromfile(mesh_file,count=3*num_vertices,dtype=np.float64).reshape((num_vertices,3))

    #Triangles
    num_triangles,=struct.unpack('Q', mesh_file.read(8))

    # dtype # 3 unsigned integers (long long) for the triangles # 2 integers for the labels
    dt=np.dtype([('triangles',np.uint64,(3,)), ('labels',np.int32,(2,))])
    t=np.fromfile(mesh_file,count=num_triangles,dtype=dt)
    mesh_file.close()

    Faces_num=t['triangles']
    Faces_labels=t['labels'] - 1
    Faces = np.hstack((Faces_num,Faces_labels))
    return(Verts, Faces.astype(int),np.array([num_vertices,num_triangles]))

def read_rec_file_num(filename,offset=-1): 
    mesh_file = open(filename, 'rb')
    Ns= []
    Verts = []
    Faces= []

    Lines=[]
    for line in mesh_file.readlines():
        L=line.decode('UTF8')
        L=L[:-1].split(' ')
        Lines.append(L)
        if len(L)==1 : 
            Ns.append(L[0])
        elif len(L)==3 : 
            Verts.append(L)
        else : 
            Faces.append(L)
    mesh_file.close()

    Faces = np.array(Faces).astype(int)
    Faces[:,[3,4]]+=offset
    Verts = np.array(Verts).astype(float)
    Ns = np.array(Ns).astype(int)
    return(Verts, Faces, Ns)





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
        

        cluster_1 = reverse_map[nodes_linked[0]]
        cluster_2 = reverse_map[nodes_linked[1]]

        if cluster_1 != cluster_2 : 
            
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


def Clean_mesh_from_seg(Seg):
    #Take a Segmentation class as entry
    
    V, Faces, Faces_idx, Nodes_linked = retrieve_mesh_multimaterial_multitracker_format(Seg.Delaunay_Graph,Seg.Map_end)
    Verts = V.copy()
    
    for i, f in enumerate(Faces): 
        if f[3]>f[4]: 
            Faces[i]=Faces[i,[0,1,2,4,3]]
            Nodes_linked[i]=Nodes_linked[i][[1,0]]
            
    
    Faces = reorient_faces(Faces,Seg,Nodes_linked)

    for i in range(len(Faces)): 
        Faces[i]=Faces[i,[0,2,1,3,4]]
            
    return(Verts,Faces)


def compute_normal_Faces(Verts,Faces):
    Pos = Verts[Faces[:,[0,1,2]]]
    Sides_1 = Pos[:,1]-Pos[:,0]
    Sides_2 = Pos[:,2]-Pos[:,1]
    Normal_faces = np.cross(Sides_1,Sides_2,axis=1)
    Norms = np.linalg.norm(Normal_faces,axis=1)#*(1+1e-8)
    Normal_faces/=(np.array([Norms]*3).transpose())
    return(Normal_faces)

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