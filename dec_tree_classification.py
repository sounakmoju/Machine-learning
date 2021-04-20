class decision_node():
    def __init__(self, column=None,threshhold=None,value=None,L_branch=None,R_branch=None):
        self.column=column
        self.threshhold=threshhold
        self.value=value
        self.L_branch=L_branch
        self.R_branch=R_branch
class decision_tree(object):
    def __init__(self, min_split=2, min_entropy=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root=None
        self.min_split=min_split
        self.min_entropy=min_entropy
        self.max_depth=max_depth
        self.gain=None
        self.loss=None
        self.leaf_value=None
    def build_tree(self,X,y,cur_depth=0):
        overall_entropy=99999
        best_criteria=None
        best_sets=None
        
        y=np.expand_dims(y,axis=1)
        samples,n_features=np.shape(X)
        Xy=np.concatenate((X,y),axis=1)
        if samples>=self.min_split and current_depth<self.max_depth:
            for i in range(n_features):
                feature_values=np.expand_dims(X[:,i])
                unique_values=np.unique(feature_values)

                for threshold in unique_values:
                    Xy1,XY2=split_data(Xy,i,threshold)
                    if len(Xy1)>0 and len(Xy2)>0:
                        entropy=calculate_overall_entropy(Xy1,Xy2)
                        if entropy<overall_entropy:
                            overall_entropy=entropy
                            best_criteria={"spilit_column"=i,"threshhold"=threshold}
                            best_sets={
                                "L_X":Xy1[:,:n_features],
                                "L_y":Xy1[:,n_features:],
                                "R_X":Xy2[:,:n_features],
                                "R_Y":Xy2[:,n_features:]
                                }
        if overall entropy>self.min_entropy:
            Left_branch=self.build_tree(best_sets["L_X"],best_sets["L_y"],cur_depth=cur_depth+1)
            Right_branch=self.build_tree(best_sets["R_X"],best_sets["R_Y"],cur_depth=cur_depth+1)
            return decision_node(column=best_criteria["spilit_column"],threshhold=best_criteria["threshhold"],L_branch=Left_branch,R_branch=Right_branch)
        else:
            leaf_value=self.leaf_value(y)
            value=self.leaf_value
            
        

