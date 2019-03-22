from scipy import sparse
import numpy as np

def densify(feature):
    return np.asarray(feature.todense())[:, 0] if sparse.issparse(feature) else feature