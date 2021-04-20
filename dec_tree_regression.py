import numpy as np
import split_data from data_mani
class Decision_node():
     def __init__(self, column=None, threshold=None,
                node_value=None, left_node=None, Right_node=None):
         self.column=column
         self.threshold=threshold
         self.node_value=node_value
         self.left_node=left_node
         self.right_node=right_node
class decisiontree(object):
    def __init__(self, min_samples_split=2,min_impurity=1e-3,
                 max_depth=99999, loss=None):
        self.root=None
        self.min_samples_split=min_samples_split
        self.min_impurity=min_impurity
        self.max_depth=max_depth
        self.loss=loss
        self.impurity_cal=None
        self._leaf_value=None
        #self._impurity_cal= None
    def fit(self,X,y,loss=None):
         self.root=self.build_tree(X,y)
         self.loss=None
    def build_tree(self, X, y, curr_depth=0):
        largest_impurity=0
        features=None
        sets=None
        y=np.expand_dims(y,axis=1)
        

        Xy=np.concatenate((X,y),axis=1)
        samples,columns=np.shape(X)
        if samples>=self.min_samples_split and curr_depth<=max_depth:
            for i in range(columns):
                feature_values=np.expand_dims(X[:,i],axis=1)
                unique_values=np.unique(feature_values)

                for threshold in unique_values:
                    Xy1,Xy2=split_data(Xy,i,threshhold)
                    if len(Xy1)>0 and len(Xy2)>0:
                        y1=Xy1[:,columns:]
                        y2=Xy2[:,columns:]

                        impurity=self._impurity_cal(y,y1,y2)
                        if impurity>largest_impurity:
                             largest_impurity=impurity
                             features={"columns":column,"threshhold":threshold}
                             sets={
                                  "l_x":Xy1[:,:columns],
                                  "l_y":Xy1[:,columns:],
                                  "r_x":Xy2[:,:columns],
                                  "r_y":Xy2[:,columns:]}
                if largest_impurity>self.min_impurity:
                     left_node=self.build_tree(sets["l_x"],sets["l_y"],curr_depth=curr_depth+1)
                     right_node=self.build_tree(sets["r_x"],sets["r_y"],curr_depth=curr_depth+1)
                     return Decision_node(column=features["columns"],threshold=features["threshold"],left_node=left_node,right_node=right_node)
                else:
                     leaf_value=self.leaf_value(y)
                     return Decision_node(value=leaf_value)
     def pre_val(self,x,tree=None):
          if tree is None:
               tree=self.root
          if tree.value is not None:
               return tree.value
          feature_value=x[tree.column]
          pre_node=tree.left_node
          if feature_value>=tree.threshhold:
               pre_node=tree.right_node
          elif feature_value==tree.threshold:
               pre_node=tree.right_node
          return self.pre_val(x,pre_node)
     def predict(self,X):
          y_pred=[self.pre_val(i) for i in range X]
          return y_pred
     class Regressiontree(decisiontree):
          def variance_reduction(self,y,y1,y2):
               var_tot=calculate_var(y)
               var_1=calculate_var(y1)
               var_2=calculate_var(y2)
               frac_1=len(y1)/len(y)
               frac_2=len(y2)/len(y)
               variance_reduction=var_tot-(frac_1*var_1+frac_2*var_2)
               return sum(varaince_reduction)
          def mean_y(self,y):
               value=np.mean(y,axis=0)
               return value
          def fit(self,X,Y):
               self._impurity_cal=self.variance_reduction
               self.leaf_value=self.mean_y
               super(Regressiontree,self).fit(X,y)
    
