from test_train_spllit,mean_squared_error,calculate_var,calculate_std_dev
class gradient_boosting(object):
     def __init__(self, no_trees, learning_rate, min_samples_split,
                 min_impurity, max_depth):
         self.no_trees=no_trees
         self.learning_rate=learning_rate
         self.min_samples_split=min_samples_split
         self.min_impurity=min_impurity
         self.max_depth=max_depth
         self.loss=mean_squared_error()

         self.trees=[]
         for_in range(no_trees):
             tree=Regressiontree(min_samples_split=self.min_samples,min_impurity=min_impurity,
                 max_depth=self.max_depth)
             self.trees.append(tree)
    def fit(self,X,y):
        y_pred=n.full(np.shape(y),np.mean(y,axis=0))
        for i in range(self.no_trees):
            gradient=self.loss.gradient(y,y_pred)
            self.trees[i].fit(X,gradient)
            update=self.trees[i].predict(X)
            y_pred-=np.multiply(self.learning_rate,update)
    def predict(self,X):
        y_pred=np.array([])
        for tree in self.trees:
            update=tree.predict(X)
            update=np.multiply(self
         
    
    
