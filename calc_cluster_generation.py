'''
calc_cluster_generation: methods for creating microcalcifications and clusters. 
Calcifications and clusters are ndarrays filled with 1s and 0s (1s indicate presence of calcification) and converted to a .RAW file.


Explanations and more details for methods from raster_geometry can be found here https://github.com/norok2/raster_geometry/blob/master/raster_geometry/raster.py. 

See cluster_generation.ipynb for example of how to create the clusters.

Methods:

* create_calc
* create_calc_rod

* create_cluster
* create_cluster_nonuniform
* create_cluster_linear
* create_cluster_rod
* create_cluster_rod_sphere
'''
# ======================================================================
# External Library Imports
import numpy as np
import random as rand
import math

#!pip install raster_geometry
from raster_geometry import * # For creating spherical and cylindrical ndarrays (https://github.com/norok2/raster_geometry/).

#!pip install bresenham
from bresenham import bresenham # For generating coordinates for linear clusters (https://github.com/encukou/bresenham).


#Creating individual calcifications
#Spherical calcifications
def create_calc(size, num_rmv_calc):
    '''
    Creates one microcalcification. 
    
    Arguments: 
        size - diameter of calcification.
        num_rmv_calc - how many voxels to remove from the outer surface.
        
    Returns:
        calc_init (ndarray) - calcification of size x size x size. 
    '''
    #Method from raster_geometry. 
    calc_init=sphere(size, size//2).astype(int)
    
    #Indexes of voxels on the outer surface of the calcification. 
    #Includes voxels on the top of the array and the side of the array.
    #num_rmv_calc specifies how many voxels are removed. 
    tops = [(ii, i, j)  for i in range(size) for ii in range(size) 
            if i==0 or i==size-1 for j in range(size)]
    sides = [(ii, i, jj) for jj in [0, size-1] for i in range(size) for ii in range(size)]
    tot=set(tops+sides)
    
    #Remove the outer surface voxels. 
    to_rmv=rand.sample(tot, int(num_rmv_calc))
    for ind in to_rmv: 
        calc_init[ind[0], ind[1], ind[2]]=0
    return calc_init

# ======================================================================

#Rod-like calcifications
def create_calc_rod(size, num_rmv_calc, smoothing=None, axis=-1, both=False):
    '''
    Creates rod-like, rectangular calcifications. 
    
    Arguments:
        size - Length of calcification's longest side.
        num_rmv_calc - how many voxels to remove from the outer surface.
        smoothing - smoothing of the border, default = None. Details found at https://github.com/norok2/raster_geometry/blob/master/raster_geometry/raster.py. 
        axis - orientation of calcification. Default axis = -1 for vertical calcifications,  
            axis = -2 for horizontal calcifications.
        both - boolean to indicate whether cluster includes both vertical and horizontal calcifications.
            Default = False.
     
    Returns:
        calc_init (ndarray) - calcification of size x size x size. 
    '''
    
    if both:
        if rand.uniform(0,1) < 0.5:
            axis=-2
            
    #Method from raster_geometry.
    calc_init=cylinder(size, size, size//3, axis=axis, smoothing=smoothing).astype(int)
    
    #Indexes of voxels on the outer surface of the calcification. 
    #Includes voxels on the top of the array and the side of the array.
    #num_rmv_calc specifies how many voxels are removed. 
    tops = [(ii, i, j)  for i in range(size) for ii in range(size) 
            if i==0 or i==size-1 for j in range(size)]
    sides = [(ii, i, jj) for jj in [0, size-1] for i in range(size) for ii in range(size)]
    tot=set(tops+sides)
    
    #Remove the outer surface voxels. 
    to_rmv=rand.sample(tot, int(num_rmv_calc))
    for ind in to_rmv: 
        calc_init[ind[0], ind[1], ind[2]]=0
    return calc_init

# ======================================================================

#Creating microcalcification clusters
#Random clusters
def create_cluster(cluster_size, num_calcs, min_calc_size=3, max_calc_size=9, weights=None, rod=False, both_shape=False):
    '''
    Creates random calcification cluster.
    Arguments:
        cluster_size - voxel length of one side of cluster. (Specification for MC-GPU: 1 vx^3 = 70 micromm.)
        num_calcs - how many individual microcalcifications located inside the cluster.
        min_calc_size - smallest microcalcification size, default = 3 voxels. 
        max_calc_size - largest microcalcification size, default = 9 voxels.
        weights (list) - distribution of sizes of the microcalcifications. From left to right, should correspond
            with the min_calc_size up to the max_calc_size, default = None. 
        rod - boolean for whether only rod-like calcifications will be placed in cluster (vertical and no smoothing), default = False.
        both_shape - boolean for whether rod-like (vertical and no smoothing) and spherical calcifications will be placed in cluster, 
            default = False.
    Returns:
        cluster_type (ndarray) - cluster with <num_calcs> spherical randomly placed calcifications. 
    '''
    if weights is None: 
        #List of sizes for individual calcifications.
        calc_size_list = [rand.randint(min_calc_size, max_calc_size) for i in range(num_calcs)]
    else:
         #List of sizes for individual calcifications based on probabilities given. 
        calc_size_list = rand.choices([i for i in range(min_calc_size, max_calc_size+1)], weights, k=num_calcs)
    
    #Create ndarray for cluster filled with 0s. 
    cluster_init= np.zeros((cluster_size, cluster_size, cluster_size))
    
    #Create list of all possible indexes for calcifications inside the cluster.
    cluster_ind=[(i, j, k) for i in range(cluster_size-max_calc_size) 
            for j in range(cluster_size-max_calc_size) for k in range(cluster_size-max_calc_size)]
    
    #Choose a set number of indexes for the calcifications specified by num_calcs. 
    calc_place_ind=rand.sample(cluster_ind, num_calcs)
    
    for calc_size, ind in zip(calc_size_list, calc_place_ind):
        if both_shape: #Rod and spherical
            
            if rand.uniform(0,1) < 0.5:
                
                #Remove number of voxels equal to calc_size squared. 
                #Ex. For calc of size 3, remove 3^2 = 9 voxels from outer surface
                calc_num_rmv = calc_size**2
                calc=create_calc_rod(calc_size, calc_num_rmv) 
            else:
                
                #Choose number for how many outer surface voxels to remove. 
                #Remove random number that is within range of 20-30% of calc_size^3.
                #Ex. For a calc of size 3, 0.2*3^3 = 0.2*27 = 5 (5.4) voxels from the outer surface will be removed. 
                calc_num_rmv = rand.randint(int(calc_size**3 * 0.2), int(calc_size**3 * 0.3))
                calc=create_calc(calc_size, calc_num_rmv) 
                
        elif rod: #Only rod
            
            #Remove number of voxels equal to calc_size squared. 
            #Ex. For calc of size 3, remove 3^2 = 9 voxels from outer surface
            calc_num_rmv = calc_size**2
            calc=create_calc_rod(calc_size, calc_num_rmv) 
            
        else: #Only spherical
            
            #Choose number for how many outer surface voxels to remove. 
            #Remove random number that is within range of 20-30% of calc_size^3.
            #Ex. For a calc of size 3, 0.2*3^3 = 0.2*27 = 5 (5.4) voxels from the outer surface will be removed. 
            calc_num_rmv = rand.randint(int(calc_size**3 * 0.2), int(calc_size**3 * 0.3))
        
            #Remove set number of voxels.
            calc=create_calc(calc_size, calc_num_rmv) 
            
        cluster_init[ind[0]:ind[0]+calc_size, ind[1]:ind[1]+calc_size, ind[2]:ind[2]+calc_size]=calc
        
    #Change ndarray type to unsigned integer of 8 bits.
    cluster_type = cluster_init.astype('uint8')
    return cluster_type

# ======================================================================

#Random non-uniform clusters
def create_cluster_nonuniform(cluster_size, num_calcs, min_calc_size=3, max_calc_size=9, weights=
                                                                   [0.2, 0.2, 0.2, 0.2, 0.03, 0.03, 0.03], rod=False, both_shape=False):
    '''
    Creates random calcification cluster with non-uniform distribution of sizes.
    Arguments:
        cluster_size - voxel length of one side of cluster. (Specification for MC-GPU: 1 vx^3 = 70 micromm.)
        num_calcs - how many individual microcalcifications located inside the cluster.
        min_calc_size - smallest microcalcification size, default = 3 voxels. 
        max_calc_size - largest microcalcification size, default = 9 voxels.
        weights (list) - distribution of sizes of the microcalcifications. From left to right, should correspond
            with the min_calc_size up to the max_calc_size. Default = [0.2, 0.2, 0.2, 0.2, 0.03, 0.03, 0.03], where
            the larger sizes, 7-9 voxels, have a probability of 10%. 
        rod - boolean for whether only rod-like calcifications will be placed in cluster (vertical and no smoothing), default = False.
        both_shape - boolean for whether rod-like (vertical and no smoothing) and spherical calcifications will be placed in cluster,
            default = False.
        
    Returns:
        cluster_type (ndarray) - cluster with <num_calcs> spherical randomly placed calcifications with non-uniform distribution 
            of sizes. 
    '''
    
    return create_cluster(cluster_size, num_calcs, min_calc_size, max_calc_size, weights, rod, both_shape)
  
# ======================================================================

#Clusters with rod-calcs with vertical and no smoothing
def create_cluster_rod(cluster_size, num_calcs, min_calc_size=3, max_calc_size=9, weights=None):
    '''
    Creates cluster with randomly placed rod-like calcifications (vertical and no smoothing). 
    Arguments:
        cluster_size - voxel length of one side of cluster. (Specification for MC-GPU: 1 vx^3 = 70 micromm.)
        num_calcs - how many individual microcalcifications located inside the cluster.
        min_calc_size - smallest microcalcification size, default = 3 voxels. 
        max_calc_size - largest microcalcification size, default = 9 voxels.
        weights (list) - distribution of sizes of the microcalcifications. From left to right, should correspond
            with the min_calc_size up to the max_calc_size, default = None. 
  
    Returns:
        cluster_type (ndarray) - cluster with <num_calcs> rod-like randomly placed calcifications. 
    '''
    
    return create_cluster(cluster_size, num_calcs, min_calc_size, max_calc_size, weights, rod=True, both_shape=False)
    

# ======================================================================

#Clusters with rod-like and spherical calcs
def create_cluster_rod_sphere(cluster_size, num_calcs, min_calc_size=3, max_calc_size=9, weights=None):
    '''
    Creates cluster with randomly placed rod-like calcifications (vertical and no smoothing). 
    Arguments:
        cluster_size - voxel length of one side of cluster. (Specification for MC-GPU: 1 vx^3 = 70 micromm.)
        num_calcs - how many individual microcalcifications located inside the cluster.
        min_calc_size - smallest microcalcification size, default = 3 voxels. 
        max_calc_size - largest microcalcification size, default = 9 voxels.
        weights (list) - distribution of sizes of the microcalcifications. From left to right, should correspond
            with the min_calc_size up to the max_calc_size, default = None. 
            
    Returns:
        cluster_type (ndarray) - cluster with <num_calcs> spherical and rod-like randomly placed calcifications. 
    '''
    return create_cluster(cluster_size, num_calcs, min_calc_size, max_calc_size, weights, rod=False, both_shape=True)

# ======================================================================

#Linear clusters
def create_cluster_linear(cluster_size, num_calcs, min_calc_size = 3, max_calc_size = 7, num_away_min = 10, num_away_max = 30):
    '''
    Creates linear cluster with spherical calcifications. 
    Arguments:
        cluster_size - voxel length of one side of cluster. (Specification for MC-GPU: 1 vx^3 = 70 micromm.)
        num_calcs - how many individual microcalcifications located inside the cluster.
        min_calc_size - smallest microcalcification size, default = 3 voxels. 
        max_calc_size - largest microcalcification size, default = 9 voxels.
        num_away_min - smallest number of pixels that calcification is away from the central line, default = 10.
        num_away_max - largest number of pixels that calcification is away from the central line, default = 30.
            
    Returns:
        cluster_type (ndarray) - cluster with <num_calcs> spherical linear placed calcifications. 
    '''
    distance=0
    length = int(math.sqrt(2) * cluster_size) - 50 #170
    x = 0
    y = 0
    xx = 0
    yy = 0
    
    #Find coordinates for a line that fits within the given cluster_size. The two end coordinates of line are (x, y) and (xx, yy).
    while distance not in range(length-10, length+10):
        x=rand.randint(max_calc_size, cluster_size-max_calc_size)
        y=rand.randint(max_calc_size, cluster_size-max_calc_size)
        xx=rand.randint(max_calc_size, cluster_size-max_calc_size)
        yy=rand.randint(max_calc_size, cluster_size-max_calc_size)
        distance=math.sqrt((x-xx)**2+(y-yy)**2)
    
    #Create list of all possible indexes for calcifications inside the cluster based on coordinates of line. 
    coordinates = list(bresenham(x, y, xx, yy))
    
    #List of sizes for individual calcifications.
    calc_size_list = [rand.randint(min_calc_size, max_calc_size) for i in range(num_calcs)]
    
    #Choose a set number of indexes for the calcifications specified by num_calcs. 
    calc_place_ind=rand.sample(coordinates, num_calcs)

    #Create ndarray for cluster filled with 0s. 
    cluster_init= np.zeros((cluster_size, cluster_size, cluster_size))
    for calc_size, indd in zip(calc_size_list, calc_place_ind):
        
        ind = (indd[0]+rand.randint(num_away_min, num_away_max)*rand.randrange(-1,2, 1), indd[1]+rand.randint(num_away_min, 
                                                                                    num_away_max)*rand.randrange(-1,2, 1))
        while not (ind[0] in range(0, cluster_size) and ind[0]+calc_size in range(0, cluster_size))                                           or not (ind[1] in range(0, cluster_size) and ind[1]+calc_size in range(0, cluster_size)):
            ind = (indd[0]+rand.randint(num_away_min, num_away_max)*rand.randrange(-1,2, 1), 
                                                       indd[1]+rand.randint(num_away_min, num_away_max)*rand.randrange(-1,2, 1))
        
        #Choose number for how many outer surface voxels to remove. 
        #Remove random number that is within range of 20-30% of calc_size^3.
        #Ex. For a calc of size 3, 0.2*3^3 = 0.2*27 = 5 (5.4) voxels from the outer surface will be removed. 
        calc_num_rmv = rand.randint(int(calc_size**3 * 0.2), int(calc_size**3 * 0.3))
        calc=create_calc(calc_size, calc_num_rmv) 

        if ind[0]+calc_size and ind[1]+calc_size < cluster_size:
            cluster_init[50:50+calc_size, ind[0]:ind[0]+calc_size, ind[1]:ind[1]+calc_size]=calc

    cluster_type = cluster_init.astype('uint8')
    return cluster_type


