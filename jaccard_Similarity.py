def extract_features(component_status_df):
    """
    Extract features from component_status_df.
    Parameters
    ----------
    component_status_df
    
    Returns
    ----------
    features: a matrix, each row is a record, each column contains the values of a certain feature
    feature_name_list: a list of headers of component_status_df
    """
    status_df = component_status_df.drop('stats_upload_id', 1)
    feature_name_list = list(status_df.columns.values)
    n = len(feature_name_list)
    features = []
    for i in range(n):
        tmp = np.array(status_df[feature_name_list[i]])
        features.append(tmp)
    return features, feature_name_list


def jaccard(f1, f2):
    """
    Computes the Jaccard metric, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    jaccard : float
        Jaccard metric returned is a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
    
    Notes
    -----
    The order of inputs for `jaccard` is irrelevant. The result will be
    identical if `f1` and `f2` are switched.
    """
    f1 = np.asarray(f1).astype(np.bool)
    f2 = np.asarray(f2).astype(np.bool)

    if f1.shape != f2.shape:
        raise ValueError("Shape mismatch: f1 and f2 must have the same shape.")

    intersection = np.logical_and(f1, f2)

    union = np.logical_or(f1, f2)

    return intersection.sum() / float(union.sum())

def build_matrix(features):
    """
    Build jaccard similarity table from features.
    Parameters
    ----------
    features: a matrix, each row is a record, each column contains the values of a certain feature
    
    Returns
    ----------
    res: jaccard similarity table
    """
    res = []
    n = len(features)
    for i in range(n):
        tmp = []
        for j in range(n):
            tmp.append(jaccard(features[i], features[j]))
        res.append(tmp)
    return res

def visualize_matrix(res, feature_name_list):
    ## add row and column names to display
    columns, rows = feature_name_list, feature_name_list
    jaccard_table = pd.DataFrame(res, index = rows, columns = columns)
    return jaccard_table

def interpret(res, feature_name_list):
    n = len(feature_name_list)
    interpretations = []
    ## pairwise relationship with risk
    for i in range(n-1):
        if res[i][-1] == 0:
            incorrect = "** Pairwise Jaccard Coefficient of " + feature_name_list[i] + " and " + feature_name_list[-1] + " is 0, which means the feature " + feature_name_list[i] + " occurs so rarely or is incorrect that it doesn't contribute to the rule prediction." 
            interpretations.append(incorrect) 
        if res[i][-1] == 1:
            defining_feature = "** Pairwise Jaccard Coefficient of " + feature_name_list[i] + " and " + feature_name_list[-1] + " is 1, meaning the feature " + feature_name_list[i] + " behaves exactly the same with the rule prediction, which is the defining feature in the rule, and other factors may be trivial"
            interpretations.append(defining_feature)
    ## pairwise relationship between features
    for i in range(n-1):
        for j in range(i, n-1):
            if i != j and res[i][j] == 1:
                count += 1
                duplicate_feature = "** Pairwise Jaccard Coefficient of " + feature_name_list[i] + " and " + feature_name_list[j] + " is 1, it means the two features behaves exactly the same, and one of them may be duplicate feature."
                interpretations.append(duplicate_feature)
    return interpretations


def main():
	## start with component_status_df
	## 1: extract features, and headers
	## 2. caluclate jaccard table
	## 3. build feature matrix, and display
	## 4. interpret the matrix	
	features, feature_name_list = extract_features(component_status_df)
	res = build_matrix(features)
	jaccard_table = visualize_matrix(res, feature_name_list)
	interpret(res, feature_name_list)
	

