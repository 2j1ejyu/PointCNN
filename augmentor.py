import torch
import torch.nn as nn
import math

#default for PointNet2
class PointAugment_simple(object):
    def __init__(self, args):
        super(PointAugment_simple, self).__init__()
        self.axis=args.rotate_axis
        self.sigma=args.jitter_sigma
        self.clip=args.jitter_clip
        self.scale_low = args.scale_low
        self.scale_high = args.scale_high
        self.shift_range = args.shift_range
        self.rotate_sigma = args.rotate_sigma
        self.shuffle = args.shuffle
        self.rotate_gaussian = args.rotate_gaussian

    def __call__(self, x):   #input is only positions (need to update if input has feat)
        rotated_data = rotate(x, self.axis, self.rotate_sigma, self.rotate_gaussian)
        jittered_data = random_scale(rotated_data, self.scale_high, self.scale_low)
        jittered_data = shift(jittered_data, self.shift_range)
        augmented_data = jitter(jittered_data, self.clip, self.sigma)
        if self.shuffle=='True':
            augmented_data = shuffle(augmented_data)
        
        return augmented_data

    
class PointAugment_simple__(object):   # don't use argparse
    def __init__(self, axis='y', sigma=0.01, clip=0.05, scale_low=0.8, scale_high=1.25, shift_range=0.1, rotate_sigma=180, rotate_gaussian=False):
        super(PointAugment_simple__ , self).__init__()
        self.axis=axis
        self.sigma=sigma
        self.clip=clip
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.shift_range = shift_range
        self.rotate_sigma = rotate_sigma
        self.rotate_gaussian = rotate_gaussian

    def __call__(self, x):
        rotated_data = rotate(x, self.axis, self.rotate_sigma, self.rotate_gaussian)
        jittered_data = random_scale(rotated_data, self.scale_high, self.scale_low)
        jittered_data = shift(jittered_data, self.shift_range)
        augmented_data = jitter(jittered_data, self.clip, self.sigma)
        augmented_data = shuffle(augmented_data)
        
        return augmented_data
    
    
def rotate(batch_data, axis, rotate_sigma, rotate_gaussian):   
    """ Randomly rotate the point clouds 
        Input:
            BxNx3 
        Return:
            BxNx3 rotated batch of point clouds
    """
    device = batch_data.device
    PI = torch.tensor([math.pi],device=device)
    rotated_data = torch.zeros(batch_data.shape, dtype=batch_data.dtype, device = device)
    for k in range(batch_data.shape[0]):
        if rotate_gaussian == 'True': 
            theta = PI*torch.randn((1*1),device= device)*(rotate_sigma/180)
        else:  # Uniform distribution, use rotate_simga as rotate range here
            theta = PI*(torch.rand((1*1),device= device)-0.5)*(rotate_sigma/90)
            
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        # axis = 'y': elevation rotation, axis = 'z': azimuth rotation
        if(axis == 'y'):
            rotation_matrix = torch.tensor([[cos_theta, 0, sin_theta],
                                        [0, 1, 0],
                                        [-sin_theta, 0, cos_theta]],device = device)
        elif(axis == 'x'):
            rotation_matrix = torch.tensor([[1, 0, 0],
                                        [0, cos_theta, -sin_theta],
                                        [0, sin_theta, cos_theta]],device = device)
        elif(axis == 'z'):
            rotation_matrix = torch.tensor([[cos_theta, -sin_theta, 0],
                                        [sin_theta, cos_theta, 0],
                                        [0, 0, 1]],device = device)
        else:
            assert True, "axis setting is wrong"
        
        kth_batch = batch_data[k, ...]
        rotated_data[k, ...] = torch.matmul(kth_batch.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def jitter(batch_data, clip, sigma):
    """ Randomly jitter points (per point)
        Gaussian N(0,sigma)
        Input:
          BxNx3 
        Return:
            BxNx3 jittered batch of point clouds
    """
    device = batch_data.device
    assert(clip > 0), "clip is 0 or less than 0"
    jittered_data = torch.clamp(sigma * torch.randn(batch_data.shape,device=device), -1*clip, clip)  #clip
    jittered_data += batch_data
    return jittered_data
    
def shuffle(batch_data):
    """ Shuffle orders of points
        Input:
            BxNx3
        Output:
            BxNx3
        Use the same shuffling idx for the entire batch.
    """
    device = batch_data.device
    rand_idx = torch.randperm(batch_data.shape[1],device=device)
    return batch_data[:,rand_idx,:]
    
def random_scale(batch_data, scale_high, scale_low):
    """ Randomly scale the point cloud. Scale is per point cloud.
    Input:
        BxNx3 array
    Return:
        BxNx3 array, scaled batch 
    """
    device = batch_data.device
    B = batch_data.shape[0]
    scales = (scale_high-scale_low)*torch.rand(B,device=device)+scale_low
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data
    
def shift(batch_data, shift_range):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
            BxNx3 array,
        Return:
            BxNx3 array, shifted batch
    """
    device = batch_data.device
    B= batch_data.shape[0]
    shifts = (2*shift_range) * torch.rand((B,3), device=device) + (-shift_range)
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data