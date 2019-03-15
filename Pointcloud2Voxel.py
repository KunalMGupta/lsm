from pyntcloud import PyntCloud
import numpy as np
import pandas as pd
import pickle
import random

vertices = []
faces = []

with open('/datasets/home/76/776/k5gupta/cmr/all_preds/time_Mar_14_16:33:42/56332360ecedaf4fb095dfb45b5ad0ce/render_0.pkl', 'rb') as f: 
    vertices, faces = pickle.load(f, encoding='latin1')
    
      
parts = 100
new_vertices = []
for f in faces:
    vs = [vertices[f[0]], vertices[f[1]], vertices[f[2]]]
    
    for i in range(parts):
        r1 = random.random()
        r2 = random.random() * (1-r1)
        r3 = 1-r1-r2
        
        new_vertices.append(vertices[f[0]]*r1 + vertices[f[1]]*r2 + vertices[f[2]]*r3)
 
#print(new_vertices)
new_vertices = np.array(new_vertices)
seg = pd.DataFrame(new_vertices)
seg.columns = ['x','y','z']
cloud = PyntCloud(seg)

voxelgrid_id = cloud.add_structure("voxelgrid", n_x=32, n_y=32, n_z=32, regular_bounding_box=True)#n_x=32, n_y=32, n_z=32, regular_bounding_box=True) #size_x=0.03125, size_y=0.03125, size_z=0.03125, regular_bounding_box=False)
new_cloud = cloud.get_sample("voxelgrid_nearest", voxelgrid_id=voxelgrid_id, as_PyntCloud=True)
voxelgrid = cloud.structures[voxelgrid_id]
print(np.sum(voxelgrid.get_feature_vector(mode="density")))
print(voxelgrid.voxel_x, voxelgrid.voxel_y, voxelgrid.voxel_z)
print(voxelgrid.voxel_x.shape, voxelgrid.voxel_y.shape, voxelgrid.voxel_z.shape, voxelgrid.voxel_n, voxelgrid.shape)
print(voxelgrid.x_y_z)

min_max_x, min_max_y, min_max_z = voxelgrid.segments
print(min_max_x, min_max_y, min_max_z )
#anky=PyntCloud.from_file("../cmr/a_objfile.ply")

#anky_cloud = anky.get_sample(
#    "mesh_random", 
#    n=20000, 
#    rgb=False, 
#    normals=False, 
#    as_PyntCloud=True)

#voxelgrid_id = anky_cloud.add_structure("voxelgrid", n_x=64, n_y=64, n_z=64)

#voxelgrid = anky_cloud.structures[voxelgrid_id]

#voxelgrid_id = cloud.add_structure("voxelgrid", size_x=0.05, size_y=0.05, size_z=0.05, regular_bounding_box=False)
#voxelgrid = cloud.structures[voxelgrid_id]


# UPSAMPLE:

# UPSAMPLE DONE#
        