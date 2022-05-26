#Sacha Ichbiah, Sept 2021
import numpy as np 

def compute_areas_faces(Mesh):
    Pos = Mesh.v[Mesh.f[:,[0,1,2]]]
    Sides = Pos-Pos[:,[2,0,1]]
    Lengths_sides = np.linalg.norm(Sides,axis = 2)
    Half_perimeters = np.sum(Lengths_sides,axis=1)/2

    Diffs = np.array([Half_perimeters]*3).transpose() - Lengths_sides
    Areas = (Half_perimeters*Diffs[:,0]*Diffs[:,1]*Diffs[:,2])**(0.5)
    for i,face in enumerate(Mesh.faces) : 
        face.area = Areas[i]

def compute_areas_cells(Mesh):
    areas=np.zeros(Mesh.n_materials)
    for face in Mesh.faces : 
        #print(areas,face.material_1,face)
        areas[face.material_1]+=face.area
        areas[face.material_2]+=face.area
    return(areas)

def compute_areas_interfaces(Mesh):
    Interfaces={}
    for face in Mesh.faces : 
        materials = (face.material_1,face.material_2)
        key = (min(materials),max(materials))
        Interfaces[key]=Interfaces.get(key,0)+face.area
    return(Interfaces)

def compute_volume_cells(Mesh):
    volumes = np.zeros(Mesh.n_materials)
    for i,face in enumerate(Mesh.faces) :
        index = Mesh.f[i,[0,1,2]]
        Coords = Mesh.v[index]
        inc = np.linalg.det(Coords)
        volumes[face.material_1]+=inc
        volumes[face.material_2]-=inc
    volumes/=6
    return(volumes)

def compute_volume_derivative_matrix(Mesh):
    
    DV = np.zeros((Mesh.n_materials,len(Mesh.v),3))
    ## DV defined such that :
    ## dV_k/dxi = DV[k,i]
    ## Matrix quite sparse as there is 0 coefficients for every vertex that does not belong to the cell k.
    for i, face in enumerate(Mesh.faces): 
        index = Mesh.f[i,[0,1,2]]
        i_x,i_y,i_z = index
        a,b = Mesh.f[i,[3,4]]
        Coords = Mesh.v[index]
        x,y,z = Coords[0],Coords[1],Coords[2]
        #print(index)

        DV[a,i_x] += np.cross(y,z)
        DV[a,i_y] += np.cross(z,x)
        DV[a,i_z] += np.cross(x,y)

        DV[b,i_x] -= np.cross(y,z)
        DV[b,i_y] -= np.cross(z,x)
        DV[b,i_z] -= np.cross(x,y)
    DV/=6

    return(DV)

def compute_area_derivative_dict(Mesh):

    T = np.array(sorted(compute_areas_interfaces(Mesh).keys()))
    kt = Mesh.n_materials+1
    Areas = {tuple(t):0 for t in T}
    DA = {tuple(t) : np.zeros((len(Mesh.v),3)) for t in T}

    #dA_m/d_xi = DA[m][xi] 
    #This is zero most of the time : the matrix is sparse

    for i, face in enumerate(Mesh.faces): 
        index = Mesh.f[i,[0,1,2]]
        i_x,i_y,i_z = index
        a,b = Mesh.f[i,[3,4]]
        key_m = a*kt + b
        Coords = Mesh.v[index]
        x,y,z = Coords[0],Coords[1],Coords[2]
        #print(x,y,z)
        Areas[(a,b)]+=np.linalg.norm(np.cross(y-x,z-x))/2

        #The vector e3 is orthogonal to the plane of the triangles, so it remains the same for all the triangles. 
        #The triangle is oriented x,y,z
        e3 = np.cross(z-y,x-z)/np.linalg.norm(np.cross(z-y,x-z))

        #for x, e1 = z-y 
        DA[(a,b)][i_x]+= np.cross(e3,z-y)/2

        #for y, e1 = x-z
        DA[(a,b)][i_y]+= np.cross(e3,x-z)/2

        #for z, e1 = y-x
        DA[(a,b)][i_z]+= np.cross(e3,y-x)/2
    return(DA)



def compute_angles_tri(Mesh,unique=True):
    ##We compute the angles at each trijunctions. If we fall onto a quadrijunction, we skip it
    
    dict_length={}
    dict_angles={}
    for edge in Mesh.half_edges : 
        if len(edge.twin)>1 : 
            face = edge.incident_face
            Faces = [face]
            sources=[edge.origin.key-edge.destination.key]
            Normals = [face.normal]
            materials = [[face.material_1,face.material_2]]
            
            for neighbor in edge.twin : 
                face_attached = Mesh.half_edges[neighbor].incident_face
                Faces.append(face_attached)
                sources.append(Mesh.half_edges[neighbor].origin.key-Mesh.half_edges[neighbor].destination.key)
                materials.append([face_attached.material_1,face_attached.material_2])
                Normals.append(face_attached.normal)
        

            regions_id = np.array(materials)
            if len(regions_id)!=3 : 
                continue
                ## If we fall onto a quadrijunction, we skip it. 

            normals = np.array(Normals).copy()
            
            if regions_id[0,0]==regions_id[1,0]:
                regions_id[1]=regions_id[1][[1,0]]
                normals[1]*=-1
            elif regions_id[0,1]==regions_id[1,1]:
                regions_id[1]=regions_id[1][[1,0]]
                normals[1]*=-1
                
            if regions_id[0,0]==regions_id[2,0]:
                regions_id[2]=regions_id[2][[1,0]]
                normals[2]*=-1
            elif regions_id[0,1]==regions_id[2,1]:
                regions_id[2]=regions_id[2][[1,0]]
                normals[2]*=-1

            pairs = [[0,1],[1,2],[2,0]]

            for i,pair in enumerate(pairs) : 
                i1,i2 = pair
                angle = np.arccos(np.dot(normals[i1],normals[i2]))

                if regions_id[i1][1]==regions_id[i2][0]:
                    e,f,g=regions_id[i1][0],regions_id[i1][1],regions_id[i2][1]
                    
                elif regions_id[i1][0]==regions_id[i2][1]:
                    e,f,g=regions_id[i2][0],regions_id[i2][1],regions_id[i1][1]

                dict_angles[(min(e,g),f,max(e,g))]=dict_angles.get((min(e,g),f,max(e,g)),[])
                dict_angles[(min(e,g),f,max(e,g))].append(angle)
                dict_length[(min(e,g),f,max(e,g))]=dict_length.get((min(e,g),f,max(e,g)),0)
                dict_length[(min(e,g),f,max(e,g))]+=(edge.length)
                if not unique : 
                    dict_angles[(min(e,g),f,max(e,g))]=dict_angles.get((min(e,g),f,max(e,g)),[])
                    dict_angles[(min(e,g),f,max(e,g))].append(angle)
                    dict_length[(min(e,g),f,max(e,g))]=dict_length.get((min(e,g),f,max(e,g)),0)
                    dict_length[(min(e,g),f,max(e,g))]+=(edge.length)

    dict_mean_angles = {}
    dict_mean_angles_deg = {}
    for key in dict_angles.keys(): 
        dict_mean_angles[key]=np.mean(dict_angles[key])
        dict_mean_angles_deg[key]=np.mean(dict_mean_angles[key]*180/np.pi)
        
    return(dict_mean_angles,dict_mean_angles_deg,dict_length)

def compute_angles_tri_and_quad(Mesh,unique=True):
    #We compute angles at both trijunctions and quadrijunctions
   
    dict_angles_tri={}
    dict_length_tri={}
    dict_angles_quad={}
    dict_length_quad={}
    dict_disconnect={}
    for edge in Mesh.half_edges : 
        if len(edge.twin)>1 : 
            face = edge.incident_face
            Faces = [face]
            sources=[edge.origin.key-edge.destination.key]
            Normals = [face.normal]
            materials = [[face.material_1,face.material_2]]
            
            for neighbor in edge.twin : 
                face_attached = Mesh.half_edges[neighbor].incident_face
                Faces.append(face_attached)
                sources.append(Mesh.half_edges[neighbor].origin.key-Mesh.half_edges[neighbor].destination.key)
                materials.append([face_attached.material_1,face_attached.material_2])
                Normals.append(face_attached.normal)
        

            regions_id = np.array(materials)
            normals = np.array(Normals).copy()
            if len(regions_id)==3 : 
            
                if regions_id[0,0]==regions_id[1,0]:
                    regions_id[1]=regions_id[1][[1,0]]
                    normals[1]*=-1
                elif regions_id[0,1]==regions_id[1,1]:
                    regions_id[1]=regions_id[1][[1,0]]
                    normals[1]*=-1

                if regions_id[0,0]==regions_id[2,0]:
                    regions_id[2]=regions_id[2][[1,0]]
                    normals[2]*=-1
                elif regions_id[0,1]==regions_id[2,1]:
                    regions_id[2]=regions_id[2][[1,0]]
                    normals[2]*=-1

                pairs = [[0,1],[1,2],[2,0]]

                for i,pair in enumerate(pairs) : 
                    i1,i2 = pair
                    angle = np.arccos(np.dot(normals[i1],normals[i2]))

                    if regions_id[i1][1]==regions_id[i2][0]:
                        e,f,g=regions_id[i1][0],regions_id[i1][1],regions_id[i2][1]

                    elif regions_id[i1][0]==regions_id[i2][1]:
                        e,f,g=regions_id[i2][0],regions_id[i2][1],regions_id[i1][1]

                    dict_angles_tri[(min(e,g),f,max(e,g))]=dict_angles_tri.get((min(e,g),f,max(e,g)),[])
                    dict_angles_tri[(min(e,g),f,max(e,g))].append(angle)
                    dict_length_tri[(min(e,g),f,max(e,g))]=dict_length_tri.get((min(e,g),f,max(e,g)),0)
                    dict_length_tri[(min(e,g),f,max(e,g))]+=(edge.length)
                    
                    if not unique : 
                        dict_angles_tri[(min(e,g),f,max(e,g))]=dict_angles_tri.get((min(e,g),f,max(e,g)),[])
                        dict_angles_tri[(min(e,g),f,max(e,g))].append(angle)
                        dict_length_tri[(min(e,g),f,max(e,g))]=dict_length_tri.get((min(e,g),f,max(e,g)),0)
                        dict_length_tri[(min(e,g),f,max(e,g))]+=(edge.length)

            elif len(regions_id)==4 : 
                
        
                args = np.argsort(regions_id[:,0])
                regions_id=regions_id[args]
                normals=normals[args]
                regions = np.unique(regions_id)
                #print(regions_id,normals)
                if len(regions)==3 : 
                    """
                    IMPORTANT TO IMPOSE THIS CONDITION HERE ONLY BECAUSE WE HAVE SOME PROBLEMS WITH OUR MESHES : THEY ARE NOT VERY REGULAR, WE DO NOT REALLY KNOW WHY.
                    THUS WE CAN HAVE SOME REGIONS WITH 4 REGIONS ID BUT ONLY 3 MATERIALS : NOT NORMAL !
                    FOR NORMAL MESHES WE HAVE TO REMOVE IT
                    """
                    #print(regions_id)
                    #print("pb")
                    continue
                #else : 
                #    print("success")

                indices_kept = [0]
                indices_removed = []
                for i in range(1,4): 
                    r1,r2 = regions_id[i]
                    if (r1 in regions_id[0]) or (r2 in regions_id[0]) : 
                        indices_kept.append(i)
                    else : 
                        indices_removed.append(i)
                indices_kept,indices_removed

                regions_id = regions_id[indices_kept+indices_removed]
                normals = normals[indices_kept+indices_removed]

                #JUST AS BEFORE BUT. THEN WE REALIGN THE 4th ROW (ie of index 3)
                if regions_id[0,0]==regions_id[1,0]:
                    regions_id[1]=regions_id[1][[1,0]]
                    normals[1]*=-1
                elif regions_id[0,1]==regions_id[1,1]:
                    regions_id[1]=regions_id[1][[1,0]]
                    normals[1]*=-1

                if regions_id[0,0]==regions_id[2,0]:
                    regions_id[2]=regions_id[2][[1,0]]
                    normals[2]*=-1
                elif regions_id[0,1]==regions_id[2,1]:
                    regions_id[2]=regions_id[2][[1,0]]
                    normals[2]*=-1

                if regions_id[1,0]==regions_id[3,0]:
                    regions_id[3]=regions_id[3][[1,0]]
                    normals[3]*=-1
                elif regions_id[1,1]==regions_id[3,1]:
                    regions_id[3]=regions_id[3][[1,0]]
                    normals[3]*=-1
        
                pairs = [[0,1],[2,0],[1,3],[2,3]]
                angles=[]
                #disconnected index(to know about the topology) : 
                #Give the index of the region from which the region of index 0 (the region of lowest number) is given : 
                #if regions is [1,2,4,6] -> gives 2 if 1-4 is disconnected as regions[2]=4
                disc_region = (set(regions)-set(np.unique(regions_id[[0,1]]))).pop()
                disc_index = np.arange(len(regions))[regions==disc_region][0]
                
                for i,pair in enumerate(pairs) : 
                    i1,i2 = pair
                    angle = np.arccos(np.dot(normals[i1],normals[i2]))
                    angles.append(angle)
                    
                    if regions_id[i1][1]==regions_id[i2][0]:
                        e,f,g=regions_id[i1][0],regions_id[i1][1],regions_id[i2][1]

                    elif regions_id[i1][0]==regions_id[i2][1]:
                        e,f,g=regions_id[i2][0],regions_id[i2][1],regions_id[i1][1]

                    #print(len(regions),regions)
                    r_index = (set(regions)-set([e,f,g])).pop()
                    
                    dict_angles_quad[(min(e,g),f,max(e,g),r_index)]=dict_angles_quad.get((min(e,g),f,max(e,g),r_index),[])
                    dict_angles_quad[(min(e,g),f,max(e,g),r_index)].append(angle)
                    dict_length_quad[(min(e,g),f,max(e,g),r_index)]=dict_length_quad.get((min(e,g),f,max(e,g),r_index),0)
                    dict_length_quad[(min(e,g),f,max(e,g),r_index)]+=(edge.length)
                    #dict_disconnect[(min(e,g),f,max(e,g),r_index)]=disc_index
                    

                    tuple_idx = tuple(sorted([min(e,g),f,max(e,g),r_index]))
                    dict_disconnect[tuple_idx]=disc_index
                    #dict_angles_quad[tuple_idx]=dict_angles_quad.get(tuple_idx,[])
                    #dict_angles_quad[tuple_idx].append(angle)
                    #dict_length_quad[tuple_idx]=dict_length_quad.get(tuple_idx,0)
                    #dict_length_quad[tuple_idx]+=(edge.length)

                    if not unique : 
                        dict_angles_quad[(min(e,g),f,max(e,g),r_index)]=dict_angles_quad.get((min(e,g),f,max(e,g),r_index),[])
                        dict_angles_quad[(min(e,g),f,max(e,g),r_index)].append(angle)
                        dict_length_quad[(min(e,g),f,max(e,g),r_index)]=dict_length_quad.get((min(e,g),f,max(e,g),r_index),0)
                        dict_length_quad[(min(e,g),f,max(e,g),r_index)]+=(edge.length)

                        dict_angles_quad[(min(e,g),f,r_index,max(e,g))]=dict_angles_quad.get((min(e,g),f,r_index,max(e,g)),[])
                        dict_angles_quad[(min(e,g),f,r_index,max(e,g))].append(angle)
                        dict_length_quad[(min(e,g),f,r_index,max(e,g))]=dict_length_quad.get((min(e,g),f,r_index,max(e,g)),0)
                        dict_length_quad[(min(e,g),f,r_index,max(e,g))]+=(edge.length)

                        dict_angles_quad[(r_index,f,min(e,g),max(e,g))]=dict_angles_quad.get((r_index,f,min(e,g),max(e,g)),[])
                        dict_angles_quad[(r_index,f,min(e,g),max(e,g))].append(angle)
                        dict_length_quad[(r_index,f,min(e,g),max(e,g))]=dict_length_quad.get((r_index,f,min(e,g),max(e,g)),0)
                        dict_length_quad[(r_index,f,min(e,g),max(e,g))]+=(edge.length)

                        dict_angles_quad[(r_index,f,max(e,g),min(e,g))]=dict_angles_quad.get((r_index,f,max(e,g),min(e,g)),[])
                        dict_angles_quad[(r_index,f,max(e,g),min(e,g))].append(angle)
                        dict_length_quad[(r_index,f,max(e,g),min(e,g))]=dict_length_quad.get((r_index,f,max(e,g),min(e,g)),0)
                        dict_length_quad[(r_index,f,max(e,g),min(e,g))]+=(edge.length)

                        dict_angles_quad[(max(e,g),f,min(e,g),r_index)]=dict_angles_quad.get((max(e,g),f,min(e,g),r_index),[])
                        dict_angles_quad[(max(e,g),f,min(e,g),r_index)].append(angle)
                        dict_length_quad[(max(e,g),f,min(e,g),r_index)]=dict_length_quad.get((max(e,g),f,min(e,g),r_index),0)
                        dict_length_quad[(max(e,g),f,min(e,g),r_index)]+=(edge.length)

                        dict_angles_quad[(max(e,g),f,r_index,min(e,g))]=dict_angles_quad.get((max(e,g),f,r_index,min(e,g)),[])
                        dict_angles_quad[(max(e,g),f,r_index,min(e,g))].append(angle)
                        dict_length_quad[(max(e,g),f,r_index,min(e,g))]=dict_length_quad.get((max(e,g),f,r_index,min(e,g)),0)
                        dict_length_quad[(max(e,g),f,r_index,min(e,g))]+=(edge.length)
   
    dict_mean_angles_tri = {}
    dict_mean_angles_deg_tri = {}
    for key in dict_angles_tri.keys(): 
        dict_mean_angles_tri[key]=np.mean(dict_angles_tri[key])
        dict_mean_angles_deg_tri[key]=np.mean(dict_mean_angles_tri[key]*180/np.pi)
        
    dict_mean_angles_quad = {}
    dict_mean_angles_deg_quad = {}
    for key in dict_angles_quad.keys(): 
        dict_mean_angles_quad[key]=np.mean(dict_angles_quad[key])
        dict_mean_angles_deg_quad[key]=np.mean(dict_mean_angles_quad[key]*180/np.pi)
        
    tuple_tri  = (dict_mean_angles_tri,dict_mean_angles_deg_tri,dict_length_tri)
    tuple_quad = (dict_mean_angles_quad,dict_mean_angles_deg_quad,dict_length_quad,dict_disconnect)
    return(tuple_tri,tuple_quad)


