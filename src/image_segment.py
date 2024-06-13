import numpy as np
from k_means import k_means

# segment image
def segment_image(img:np.ndarray, k:int, epochs:int, stride:int=1, c:np.ndarray=None, tol:float=1e-4, verbose:bool=True) :
    '''
    img : (row, column, depth)
    range of element should be [0.0, 1.0]
    '''
    
    # when image is grayscale
    if len(img.shape) == 2 :
        img = img[:,:,np.newaxis]
    
    img_shape = img.shape
    depth = img_shape[2]
    
    c, label = k_means(img[::stride, ::stride, :].reshape(-1, depth), k, epochs, c, tol, verbose)
    # c_np : (k, depth), label : (pixel num / stride^2)
    
    if stride > 1 :
        x_ext = img.reshape(-1,depth)[:,np.newaxis,:].repeat(k, axis=1) # (-1, k, depth)
        idx_vec = ((x_ext - c[np.newaxis,:,:])**2).sum(2).argmin(1)
        label = idx_vec

    out = c[label].reshape(img_shape) # c_np[label] : (pixel num, depth)
    return out, c