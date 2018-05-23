from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge


class RegressorsSelection:
    
    def PartialLeastSquare(X_train,Y_train,n_components):
        pls2 = PLSRegression(n_components=n_components)
        pls2.fit(X_train,Y_train)
    
    def Lasso(X_train,Y_train,alpha):
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train,Y_train)
    
    def Ridge(X_train,Y_train,alpha):
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train,Y_train)
        
    def Principal(X_train,n_components):
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        
        