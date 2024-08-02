import numpy as np
from evtk.hl import pointsToVTK

def write_geoatt_vtk3(filename, pts, **attributes):

    x    = np.ascontiguousarray(pts[:,0])
    y    = np.ascontiguousarray(pts[:,1])
    z    = np.ascontiguousarray(pts[:,2])

    data = {}

    for k, v in attributes.items() :
        data[k] = np.ascontiguousarray(v) 

    pointsToVTK(filename, x, y, z, data = data)

def write_geoatt_vtk3_prob(filename, pts, probs):

    x    = np.ascontiguousarray(pts[:,0])
    y    = np.ascontiguousarray(pts[:,1])
    z    = np.ascontiguousarray(pts[:,2])
    probs = np.array(probs).astype(np.float32) # Assure probs is np array
    n_classes = probs.shape[1]

    data = {}

    for i in range(n_classes):
        data[f'class{i+1}'] = np.ascontiguousarray(probs[:, i])

    data[f'predictedClass'] = np.argmax(probs, axis=1)
    
    pointsToVTK(filename, x, y, z, data = data)
