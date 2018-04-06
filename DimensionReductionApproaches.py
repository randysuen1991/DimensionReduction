import numpy as np


def CenteringDecorator(fun):
    def decofun(**kwargs):
        X_train = kwargs.get('X_train')
        shape = [i for i in X_train.shape[1:]]
        shape.insert(0,1)
        X_mean = np.mean(X_train,axis=0)
        X_mean = np.reshape(X_mean,newshape=shape)
        return fun(X_train-X_mean,kwargs.get('Y_train'))
        
    return decofun




def TotalCentered(X):
    shape = [i for i in X.shape[1:]]
    shape.insert(0,1)
    X_mean = np.mean(X,axis=0)
    X_mean = np.reshape(X_mean,newshape=shape)
    return X - X_mean


def WithinGroupMeanCentered(X,Y):
    Y_ravel = Y.ravel()
    shape = [i for i in X.shape[1:]]
    shape.insert(0,1)
    for i in range(int(max(Y)[0])):
        inds = np.where(Y_ravel == i + 1)[0]
        X_group = X[inds,:]            
        within_group_mean = np.mean(X_group,axis=0)
        within_group_mean = np.reshape(within_group_mean,newshape=shape)
        within_group_mean_centered = X_group - within_group_mean
        if i == 0 :
            within_groups_mean_centered = within_group_mean_centered
        else :
            within_groups_mean_centered = np.concatenate((within_groups_mean_centered,within_group_mean_centered),axis=0)
    
    return within_groups_mean_centered



def BetweenGroupMeanCentered(X,Y):
    Y_ravel = Y.ravel()
    n = X.shape[0]
    shape = [i for i in X.shape[1:]]
    shape.insert(0,1)
    for i in range(int(max(Y)[0])):
        inds = np.where(Y_ravel == i + 1)[0]
        X_group = X[inds,:]
        n_sam_sub = len(X_group)            
        between_group_mean = np.mean(X_group,axis=0)
        between_group_mean = np.reshape(between_group_mean,newshape=shape)
        between_group_mean_centered = (between_group_mean - np.mean(X,axis=0)) * np.sqrt(n_sam_sub/n)
        if i == 0 :
            between_groups_mean_centered = between_group_mean_centered 
        else :
            between_groups_mean_centered = np.concatenate((between_groups_mean_centered,between_group_mean_centered),axis=0)
    
    return between_groups_mean_centered

class DimensionReduction():
    
    @CenteringDecorator
    def PCA(X_train,**kwargs):
        p = np.linalg.matrix_rank(X_train)
        k = kwargs.get('q',p)
        _, _, V_t = np.linalg.svd(X_train,full_matrices=False)
        V = np.transpose(V_t)
        try:
            linear_subspace = V[:,0:k]
        except IndexError:
            linear_subspace = V[:,0:p]
        finally :
            return linear_subspace
            
    
class LinearDiscriminant(DimensionReduction):
    
    @CenteringDecorator
    def FLDA(X_train,Y_train,**kwargs):
        if X_train.shape[1] > X_train.shape[0]:
            raise ValueError('The dimension of the data should not be larger than the sample size of the data.')
        
        between_groups_mean_centered = BetweenGroupMeanCentered(X_train,Y_train)
        within_groups_mean_centered = WithinGroupMeanCentered(X_train,Y_train)
        
        between_matrix = np.matmul(np.transpose(between_groups_mean_centered),between_groups_mean_centered)
        within_matrix = np.matmul(np.transpose(within_groups_mean_centered),within_groups_mean_centered)
        
        target_matrix =np.matmul(np.linalg.inv(within_matrix),between_matrix)
        _, V = np.linalg.eig(target_matrix)
        
        return V
    
    @CenteringDecorator
    def FFLDA(X_train,Y_train,**kwargs):
        
        between_groups_mean_centered = BetweenGroupMeanCentered(X_train,Y_train)
        within_groups_mean_centered = WithinGroupMeanCentered(X_train,Y_train)
        
        r = np.linalg.matrix_rank(within_groups_mean_centered)
        
        _, _, V_t = np.linalg.svd(X_train,full_matrices=False)
        V_pre = np.transpose(V_t)[:,0:r]
        
        between_groups_mean_centered_proj = np.matmul(between_groups_mean_centered,V_pre)
        within_groups_mean_centered_proj = np.matmul(within_groups_mean_centered,V_pre)
        
        between_matrix = np.matmul(np.transpose(between_groups_mean_centered_proj),between_groups_mean_centered_proj)
        within_matrix = np.matmul(np.transpose(within_groups_mean_centered_proj),within_groups_mean_centered_proj)
        
        target_matrix =np.matmul(np.linalg.inv(within_matrix),between_matrix)
        _, V = np.linalg.eig(target_matrix)
        
        return np.matmul(V_pre, V)
    
    @CenteringDecorator
    def NLDA(X_train,Y_train,**kwargs):
        N = X_train.shape[0]
        
        _, _, V_t = np.linalg.svd(X_train,full_matrices=False)
        V_pre = np.transpose(V_t)[:,0:N-1]
        X_train = np.matmul(X_train,V_pre)
        
        
        within_groups_mean_centered = WithinGroupMeanCentered(X_train,Y_train)
        between_groups_mean_centered = BetweenGroupMeanCentered(X_train,Y_train)
        
        
        r = np.linalg.matrix_rank(within_groups_mean_centered)
        _, _, V_t = np.linalg.svd(within_groups_mean_centered)
        V = np.transpose(V_t)
        Q = V[:,r:]
        
        r = np.linalg.matrix_rank(between_groups_mean_centered)
        groups_mean_centered_proj = np.matmul(between_groups_mean_centered,Q)
        
        _, _, U_t = np.linalg.svd(groups_mean_centered_proj)
        U = np.transpose(U_t)[:,0:r]
        
        linear_subspace = np.matmul(Q,U)
        linear_subspace = np.matmul(V_pre,linear_subspace)
        
        return linear_subspace
    
    @CenteringDecorator
    def PIRE(X_train,Y_train,**kwargs):
        q = kwargs.get('q',3)
        between_groups_mean_centered = BetweenGroupMeanCentered(X_train,Y_train)
        
        r = np.linalg.matrix_rank(between_groups_mean_centered)
        _, _, V_t = np.linalg.svd(between_groups_mean_centered,full_matrices=False)
        V = np.transpose(V_t)[:,0:r]
        Rq = V
        append = V
        for i in range(q-1):
            append = np.matmul(np.matmul(np.transpose(X_train),X_train),append)
            Rq = np.concatenate((Rq,append),axis=1)
        # To avoid computational problem, we normalize the column vectors
        for i in range(Rq.shape[1]):
            Rq[:,i] = Rq[:,i] / np.linalg.norm(Rq[:,i])
        
        inv_half = np.matmul(np.transpose(Rq),np.transpose(X_train))
        inv = np.matmul(inv_half,np.transpose(inv_half))
        inv = np.linalg.pinv(inv)
        
        linear_subspace = np.matmul(Rq,inv)
        linear_subspace = np.matmul(linear_subspace,np.transpose(Rq))
        linear_subspace = np.matmul(linear_subspace,V)
        linear_subspace, _ = np.linalg.qr(linear_subspace)
        
        return linear_subspace
    
    
    @CenteringDecorator
    def DRLDA(X_train,Y_train,**kwargs):
        
        between_groups_mean_centered = BetweenGroupMeanCentered(X_train,Y_train)
        within_groups_mean_centered = WithinGroupMeanCentered(X_train,Y_train)
        
        
        pre_r = np.linalg.matrix_rank(X_train)
        _, _, V_t = np.linalg.svd(a=X_train,full_matrices=False)
        V = np.transpose(V_t)[:,0:pre_r]
        
        
        between_groups_mean_centered_proj = np.matmul(between_groups_mean_centered,V)
        within_groups_mean_centered_proj = np.matmul(within_groups_mean_centered,V)
        
        between_matrix = np.matmul(np.transpose(between_groups_mean_centered_proj),between_groups_mean_centered_proj)
        within_matrix = np.matmul(np.transpose(within_groups_mean_centered_proj),within_groups_mean_centered_proj)
        
        within_matrix_pinv = np.linalg.pinv(within_matrix)
        _, S, _ = np.linalg.svd(np.matmul(within_matrix_pinv,between_matrix))
        evalue = S[0]
        
        _, S, _ = np.linalg.svd(between_matrix/evalue - within_matrix)
        alpha = S[0]
        
        r = np.linalg.matrix_rank(between_groups_mean_centered)
        matrix = np.matrix(within_matrix_pinv + alpha * np.eye(N=pre_r))
        inv_matrix = np.linalg.inv(matrix)
        target = np.matmul(inv_matrix,between_matrix)
        U, _ = np.linalg.qr(target)
        linear_subspace = np.matmul(V,U[:,0:r])
        return linear_subspace
    
    
        
        
        
        
class MultilinearReduction(DimensionReduction):
    @CenteringDecorator
    def MPCA(X_train,input_shape,p_tilde,q_tilde,**kwargs):
        T = kwargs.get('T',10)
        eps = kwargs.get('eps',0.001)
        n = X_train.shape[0]
        p = input_shape[1]
        q = input_shape[2]
        X_train_imgs = np.reshape(X_train,newshape=(n,p,q))
        
        X_train_imgs_mul = np.matmul(np.transpose(X_train_imgs),X_train_imgs)
        
        
        while True :
            pass
        
        
        
