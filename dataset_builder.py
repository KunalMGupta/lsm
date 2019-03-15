from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import pickle
mypath = 'cars_test'
car_instances = [f for f in listdir(mypath) if not isfile(join(mypath,f))]

def quat2rot(quat):
    a = quat[0]
    b = quat[1]
    c = quat[2]
    d = quat[3]
    R = np.array([[a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
                  [2 * b * c + 2 * a * d, a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * c * d - 2 * a * b],
                  [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a ** 2 - b ** 2 - c ** 2 + d ** 2]])
    return R.tolist()

dataset = {}
sfm = {}
rel_path = []
mask = []
index = 0
index_list = []
bbox = {'x1':0, 'y1':0, 'x2': 224, 'y2': 224}
bbox_list = []
rot_list = []
scale_list = []
trans_list = []
for car in car_instances:
    for i in range(20):
        dataset[index] = {'rel_path': join(join(mypath,car),'render_'+str(i)+'.png'), 'mask': (plt.imread(join(join(mypath,car),'depth_'+str(i)+'.png'))<1).tolist(),
                          'id': index, 'bbox': bbox}
        sfm[index] = {'rot': sio.loadmat(join(join(mypath,car),'camera_'+str(i)+'.mat'))['quat'][0], 'scale': 420,
                      'trans': sio.loadmat(join(join(mypath,car),'camera_'+str(i)+'.mat'))['pos'][0]}
        index +=1

images = {'images':dataset}
sfm = {'sfm_anno':sfm}
file = open('dataset.pkl','wb')
pickle.dump(images,file,2)
file.close()

file = open('param.pkl','wb')
pickle.dump(sfm,file,2)
file.close()


file = open('dataset.pkl','rb')
test1 = pickle.load(file)
file.close()

file = open('param.pkl','rb')
test2 = pickle.load(file)
file.close()

print(test1['images'][20]['bbox'])

print(test2['sfm_anno'][20]['rot'])




