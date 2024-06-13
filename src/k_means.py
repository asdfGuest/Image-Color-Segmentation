import numpy as np

def k_means(x_np:np.ndarray, k:int, epochs:int, c:np.ndarray=None, tol:float=1e-4, verbose:bool=True) :
    '''
    x_np : (dataset, dim)
    '''

    # initalize
    n = x_np.shape[0]
    x_ext = x_np[:,np.newaxis,:].repeat(k, axis=1) # (dataset, k, dim)
    if c is None :
        c = x_np[np.random.choice(range(n), size=k, replace=False)] # (k, dim)
    
    # distance metric
    # v1, v2 : (dataset, k, dim)
    metric = lambda v1, v2 : ((v1 - v2)**2).sum(2)

    # train
    last_loss = None
    for epoch in range(1, epochs + 1) :
        # set cluster
        dis_mat = metric(x_ext, c[np.newaxis,:,:]) # (dataset, k)
        idx_vec = dis_mat.argmin(1)
        
        # adjust cluster centroid
        for cidx in range(k) :
            temp = x_np[idx_vec == cidx]
            c[cidx] = temp.mean(0) if temp.size > 0 else c[cidx]

        # loss
        loss = dis_mat.min(1).mean()

        if verbose :
            print("epoch %d/%d   loss %f"%(epoch, epochs, loss))
        
        # early break
        if (last_loss != None and last_loss - loss <= tol) :
            break
        last_loss = loss

    return c, idx_vec