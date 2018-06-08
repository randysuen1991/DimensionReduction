import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def CenteringDecorator(fun):
    def decofun(**kwargs):
        X_train = kwargs.get('X_train')
        
        if type(X_train) == pd.DataFrame :
            X_train = X_train.values
            
        shape = [i for i in X_train.shape[1:]]
        shape.insert(0,1)
        X_mean = np.mean(X_train,axis=0)
        
        X_mean = np.reshape(X_mean,newshape=shape)
        return fun(X_train = X_train - X_mean,
                   Y_train = kwargs.get('Y_train', None),
                   input_shape = kwargs.get('input_shape', None),
                   p_tilde = kwargs.get('p_tilde', None), 
                   q_tilde = kwargs.get('q_tilde', None),
                   n_components = kwargs.get('n_components'),
                   plot = kwargs.get('plot',False),
                   num_each_side = kwargs.get('num_each_side',None),
                   both_sides = kwargs.get('both_sides',None),
                   abs = kwargs.get('abs',None))
        
    return decofun

def PCDecorator(fun):
    
    def decofun(**kwargs):
        X_train = kwargs.get('X_train')
        r = np.linalg.matrix_rank(X_train)
        _, _, V_t = np.linalg.svd(X_train,full_matrices=False)
        V = np.transpose(V_t)[:,0:r]
        X_train = np.matmul(X_train,V)
        return np.matmul(V, fun(X_train = X_train,
                                Y_train = kwargs.get('Y_train',None)))
    
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
    for i in range(int(max(Y_ravel))):
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
    for i in range(int(max(Y_ravel))):
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
        k = kwargs.get('n_components',p)
        _, s, V_t = np.linalg.svd(X_train,full_matrices=False)
        V = np.transpose(V_t)
        if kwargs.get('plot',False):
            plt.plot(s**2)
        try:
            linear_subspace = V[:,0:k]
        except IndexError:
            linear_subspace = V[:,0:p]
        finally :
            return linear_subspace, s**2
            
    
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
        s, V = np.linalg.eig(target_matrix)
        Y_uniq = np.unique(Y_train)
        r = len(Y_uniq) - 1
        return V[:,0:r]
    
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
    
    
    @PCDecorator
    @CenteringDecorator
    def NLDA(X_train,Y_train,**kwargs):
        
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
        
        return linear_subspace
    
    
    @PCDecorator
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
    
    
    @PCDecorator
    @CenteringDecorator
    def DRLDA(X_train,Y_train,**kwargs):
        
        between_groups_mean_centered = BetweenGroupMeanCentered(X_train,Y_train)
        within_groups_mean_centered = WithinGroupMeanCentered(X_train,Y_train)
        
        
        between_matrix = np.matmul(np.transpose(between_groups_mean_centered),between_groups_mean_centered)
        within_matrix = np.matmul(np.transpose(within_groups_mean_centered),within_groups_mean_centered)
        
        within_matrix_pinv = np.linalg.pinv(within_matrix)
        _, S, _ = np.linalg.svd(np.matmul(within_matrix_pinv,between_matrix))
        evalue = S[0]
        
        _, S, _ = np.linalg.svd(between_matrix/evalue - within_matrix)
        alpha = S[0]
        
        r = np.linalg.matrix_rank(between_groups_mean_centered)
        dim = X_train.shape[1]
        matrix = np.matrix(within_matrix_pinv + alpha * np.eye(N=dim))
        inv_matrix = np.linalg.inv(matrix)
        target = np.matmul(inv_matrix,between_matrix)
        U, _ = np.linalg.qr(target)
        return U[:,0:r]
        
        
        
        
class MultilinearReduction(DimensionReduction):
    
    
    
    @classmethod
    def TensorProject(cls,X_train,A,B):
        N = X_train.shape[0]
        X_train_proj = np.zeros(shape=(X_train.shape[0],A.shape[1],B.shape[1],X_train.shape[3]))
        for i in range(N):
            X_train_proj[i,:,:,0] = np.matmul(
                                    np.matmul(np.transpose(A),X_train[i,:,:,0]),
                                    B)
        return X_train_proj
    
    
    
    
    @CenteringDecorator
    def MPCA(X_train,input_shape,p_tilde,q_tilde,**kwargs):
        return MultilinearReduction.GLRAM(X_train=X_train,input_shape=input_shape,p_tilde=p_tilde,q_tilde=q_tilde)
    
        
    
    @CenteringDecorator
    def MSIR(X_train,Y_train,input_shape,p_tilde,q_tilde,**kwargs):
        
        
        # X should be a tensor with shape = (no.sample,height,width,no.channel)
        def TransformedSumGroupMeanA(X,Y,A):
            Y_ravel = Y.ravel()
            n = X.shape[0]
            shape = (1,X.shape[1],X.shape[2],X.shape[3])
            sum_group_mean = np.zeros(shape=shape)
            for i in range(int(max(Y_ravel)[0])):
                inds = np.where(Y_ravel == i + 1)[0]
                X_group = X[inds,:,:,:]
                n_sam_sub = X_group.shape[0]
                group_mean = (n_sam_sub/n) * np.matmul(np.transpose(np.mean(X_group,axis=0)),A) 
                sum_group_mean += np.matmul(group_mean,np.transpose(group_mean))
            
            return sum_group_mean
        
        def TransformedSumGroupMeanB(X,Y,B):
            Y_ravel = Y.ravel()
            n = X.shape[0]
            shape = (1,X.shape[1],X.shape[2],X.shape[3])
            sum_group_mean = np.zeros(shape=shape)
            for i in range(int(max(Y_ravel)[0])):
                inds = np.where(Y_ravel == i + 1)[0]
                X_group = X[inds,:,:,:]
                n_sam_sub = X_group.shape[0]
                group_mean = (n_sam_sub/n) * np.matmul(np.mean(X_group,axis=0),A) 
                sum_group_mean += np.matmul(group_mean,np.transpose(group_mean))
            
            return sum_group_mean
        
        
        N = X_train.shape[0]
        p = input_shape[0]
        q = input_shape[1]
        
        
        if kwargs.get('vectors',False):
            X_train = np.reshape(X_train,newshape=(N,p,q))
        
        
        A = np.eye(p)
        
        rmsre1 = 1
        d = 1
        sg = 1
        dc = 10**(-4)
        while True :
            
            A0 = A
            rmsre0 = rmsre1
            
            sgmA = TransformedSumGroupMeanA(X_train,Y_train,A0)
            U, _, _ = np.linalg.svd(sgmA,full_matrices=False)
            B1 = U[0:q_tilde]
            
            sgmB = TransformedSumGroupMeanB(X_train,Y_train,B)
            U, _, _ = np.linalg.svd(sgmB,full_matrices=False)
            A1 = U[0:p_tilde]
            
            for i in range(N):
                A = np.matmul(A1,np.transpose(A1))
                B = np.matmul(B1,np.transpose(B1))
                AIB = np.matmul(A,np.matmul(X_train[i,:,:,0],B))
                SQ_DIFF = (X_train[i,:,:,0] - AIB)**2
                rmsre1 += np.sum(SQ_DIFF)
                
            rmsre1 = (rmsre1/N)**(0.5)
            
            d = np.abs(rmsre0-rmsre1) / rmsre0
            sg += 1
        
            if d < dc or sg>30:
                break
            
            
    def GLRAM(X_train,input_shape,p_tilde,q_tilde,**kwargs):
        
        N = X_train.shape[0]
        p = input_shape[0]
        q = input_shape[1]
        
        
        if kwargs.get('vectors',False):
            X_train = np.reshape(X_train,newshape=(N,p,q))
        
        A1 = np.random.uniform(low=-1,high=1,size=(p,p_tilde))
        
        rmsre1 = 1
        d = 1
        sg = 1
        dc = 10**(-4)
        
        while True :
            
            A0 = A1
            rmsre0 = rmsre1
            
            M_B = np.zeros(shape=(q,q))
            partial = np.matmul(A0,np.transpose(A0))
            for iter2 in range(N):
                M_B += np.matmul(np.transpose(X_train[iter2,:,:,0]),
                                 np.matmul(partial,X_train[iter2,:,:,0]))
            U, _, _ = np.linalg.svd(M_B,full_matrices=False)
            B1 = U[:,0:q_tilde]
            
            
            M_A = np.zeros(shape=(p,p))
            partial = np.matmul(B1,np.transpose(B1))
            for iter2 in range(N):
                M_A += np.matmul(np.matmul(X_train[iter2,:,:,0],partial),
                                 np.transpose(X_train[iter2,:,:,0]))
            U, _, _ = np.linalg.svd(M_A,full_matrices=False)
            A1 = U[:,0:p_tilde]
            
            for i in range(N):
                A = np.matmul(A1,np.transpose(A1))
                B = np.matmul(B1,np.transpose(B1))
                AIB = np.matmul(A,np.matmul(X_train[i,:,:,0],B))
                SQ_DIFF = (X_train[i,:,:,0] - AIB)**2
                rmsre1 += np.sum(SQ_DIFF)
                
            rmsre1 = (rmsre1/N)**(0.5)
            
            d = np.abs(rmsre0-rmsre1) / rmsre0
            sg += 1
        
            if d < dc or sg>30:
                break
        
        U, _, _ = np.linalg.svd(M_A/N,full_matrices=False)
        A = U[:,0:p_tilde]
        U, _, _ = np.linalg.svd(M_B/N,full_matrices=False)
        B = U[:,0:q_tilde]
        
        return A, B
            
