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


def fractions_to_common_denom(fractions):
    """Find a common denominator and the associated numerators of a list 
    of fractions.

    Parameters
    ----------
    fractions : list of Fraction


    Returns
    -------
    numerators : ndarray
    com_denom : int    

    """

    denoms = [i.denominator for i in fractions]
    com_denom = max(denoms)

    for i in denoms:
        if com_denom % i != 0:
            com_denom *= i

    numerators = np.array([i.numerator * int(com_denom / i.denominator)
                           for idx, i in enumerate(fractions)])

    return numerators, com_denom


def zeropad(num, largest):
    """Return a zero-padded string of a number, given the largest number.

    TODO: want to support floating-point numbers as well? Or rename function
    accordingly.

    Parameters
    ----------
    num : int
        The number to be formatted with zeros padding on the left.
    largest : int
        The number that determines the number of zeros to pad with.

    Returns
    -------
    padded : str
        The original number, `num`, formatted as a string with zeros added
        on the left.

    """

    num_digits = len('{:.0f}'.format(largest))
    padded = '{0:0{width}}'.format(num, width=num_digits)

    return padded


def get_atom_species_jmol_colours(jmol_colours_path):
    'Get RGB values of JMOL atom species colours.'
    cols = {}
    with jmol_colours_path.open() as handle:
        for ln in handle:
            ln_s = ln.strip()
            _, species, rgb_str = ln_s.split()
            rgb = [int(i) for i in rgb_str[1:-1].split(',')]
            cols.update({
                species: rgb
            })
    return cols
