def test_train_split(X,Y,train_size):
    split_i=round((len(X)*train_size))
    X_train,X_test=X[:split_i],X[split_i:]
    y_train,y_test=Y[:split_i],Y[split_i:]

    return X_train,x_test,y_train,y_test
def split_data(X,split_column,split_value):
    split_data=X[:,split_column]

    X_L=np.array(X[split_data<=split_value])
    X_R=np.array(X[split_data>split_value])

    return np.array([X_L,X_R])
def calculate_entropy(data):
    label_column=data[:,-1]
    _, counts=np.unique(label_column,return_counts=True)
    probablities=counts/sum(counts)
    entropy=sum(probablities*-np.log2(probabilities))
    return entropy
def accuracy_score(y_true,y_pred):
    accuracy=np.sum(y_true-y_pred,axis=0)/len(y_true)
    return accuracy
def calculate_overall_entropy(d_below,d_above):
    n=len(d_below)+len(d_above)
    pd_below=len(d_below)/n
    pd_above=len(d_above)/n
    overall_entropy=(pd_below*calculate_entropy(d_below)+pd_above*calculate_entropy(d_above))
    return overall_entropy
def mean_squared_error(y_true,y_pred):
    mse=np.mean(np.power(y_true-y_pred,2))
    return mse
def calculate_var(X):
    mean=np.ones(np.shape(X))*X.mean(0)
    n_samples=np.shape(X)[0]
    variance=(1/n_samples)*np.daig((X-mean).T.dot(X-mean)
     return varaiance
def calculate_std_dev(X):
    std_dev=np.sqrt(calculate_var(X))
    return std_dev
