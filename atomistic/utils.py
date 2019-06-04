"""`atomistic.utils.py`"""

import numpy as np


def get_column_vector(arr, dim=3):
    """Return a column vector from a list or 1D numpy array.

    Parameters
    ----------
    arr : list or 1D ndarray
        Input list or 1D array that is to be converted as a column vector.

    Returns
    -------
    col_vec : ndarray of shape (`dim`, 1)
        Input array as a column vector.

    """

    col_vec = np.array(arr).squeeze().reshape((dim, 1))
    return col_vec


def combination_idx(*seq):
    """
    Find the indices of unique combinations of elements in equal-length
    ordered sequences.

    Parameters
    ----------
    seq : one or more sequences
        All sequences must be of the same length. These may be lists, tuples, 
        strings or ndarrays etc.

    Returns
    -------
    tuple of (list of ndarray, ndarray)
        The list is the unique combinatons (as Numpy object arrays) and the 
        second is the indices for a given combindation.

    TODO: add some examples so it's easy to see what this does!

    """

    # Validation
    seq_len = -1
    msg = 'All sequences must have the same length.'
    for s in seq:
        if seq_len == -1:
            seq_len = len(s)
        else:
            if len(s) != seq_len:
                raise ValueError(msg)

    combined_str = np.vstack(seq)
    combined_obj = np.vstack([np.array(i, dtype=object) for i in seq])

    u, uind, uinv = np.unique(combined_str, axis=1,
                              return_index=True, return_inverse=True)

    ret_k = []
    ret_idx = []
    for i in range(u.shape[1]):

        ret_k.append(combined_obj[:, uind[i]])
        ret_idx.append(np.where(uinv == i)[0])

    return ret_k, ret_idx


def check_indices(seq, seq_idx):
    """
    Given a sequence (e.g. list, tuple, ndarray) which is indexed by another,
    check the indices are sensible.

    Parameters
    ----------
    seq : sequence
    seq_idx : sequence of int

    """

    # Check: minimum index is greater than zero
    if min(seq_idx) < 0:
        raise IndexError('Found index < 0.')

    # Check maximum index is equal to length of sequence - 1
    if max(seq_idx) > len(seq) - 1:
        raise IndexError('Found index larger than seqence length.')
