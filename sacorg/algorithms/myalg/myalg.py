from sacorg.utils import *
from sacorg.algorithms.isgraphical import *


def xsi_tilde(deg_seq, s, value):
    """
    Subtract 1 from 's' elements of deg_seq, which equal to value by starting from the end of the sequence deg_seq
    :param deg_seq: degree sequence
    :param s: number of elements
    :param value:
    :return d: if deg_seq is non-increasing sequence, than the output sequence d is also in non-increasing order
    """
    d = deg_seq.copy()
    count = 0
    inx = 0
    while count < s:
        if d[-1 - inx] == value:
            d[-1 - inx] -= 1
            count += 1
        inx += 1

    return d


def convert2binary(deg_seq, matrix_s):
    """
    Converts a given counts matrix to a binary matrix
    :param deg_seq: degree sequence
    :param matrix_s:
    :return binary_matrix: a binary matrix with row and column sums equal to deg_seq
    """
    n = len(deg_seq)
    a = max(deg_seq)

    # Copy the column sum q
    d = deg_seq.copy()

    binary_matrix = np.zeros((n, n), dtype=np.int)

    for i in range(n):
        # if the table is not completely filled
        if np.sum(d) > 0:
            # Find the indices having the highest degree greater than zero
            row_indices = [inx for inx in range(n) if d[inx] == max(d) and max(d) > 0]
            # Uniformly choose a row index at random
            row = np.random.choice(row_indices, 1, replace=False)

            for val in range(1, a + 1):
                col_indices = [inx for inx in range(n) if d[inx] == val and row != inx]
                if len(col_indices) > 0:
                    chosen_indices = np.random.choice(col_indices, matrix_s[i][val - 1], replace=False)

                    binary_matrix[row, chosen_indices] = 1
                    d[chosen_indices] -= 1

            d[row] = 0

    binary_matrix = binary_matrix + binary_matrix.T

    return binary_matrix


def kappa(d, a, i, j, row_sum, submatrix_count_values):
    """
    Recursive function to compute the number of simple graphs realizing the degree sequence d
    :param d: degree sequence
    :param a: maximum element of the initial degree sequence d^0
    :param i: row index from 0 to n-1
    :param j: column index from 0 to a-1
    :param row_sum: Sum of chosen elements up to j
    :param submatrix_count_values:
    :return: Number of simple graphs realizing the degree sequence d
    """

    total = 0
    if np.sum(d) == 0:
        return 1
    else:

        if d[i] == row_sum:
            # Set d[i] to 0
            d[i] = 0
            # Store submatrix counts to avoid from computing them every time
            count_inx = tuple(vector_of_counts(d[i + 1:], dim=a))
            if count_inx in submatrix_count_values:
                return submatrix_count_values[count_inx]
            else:
                total = kappa(d, a, i + 1, 0, 0, submatrix_count_values)
                submatrix_count_values[count_inx] = total
                return total

        else:
            conj_r = np.append(conjugate(d[i + 1:], a), 0)
            conj_z = np.append(conjugate(d[i + j + 1:], a), 0)

            # Construct the vector of counts, r, from conjugate partition of q
            r = np.asarray([conj_r[inx - 1] - conj_r[inx] for inx in np.arange(1, len(conj_r))] + [conj_r[-1]])

            # Determine lower bound for the entry in ith row and jth column
            lower_bound = max(0, d[i] - row_sum - conj_r[j + 1])
            # Determine upper bound for the entry in ith row and jth column
            gale_ryser_condition = np.sum(conj_z[0:j + 1]) - np.sum(d[i + 1:(i + 1) + j + 1]) + ((j + 1) * (j + 2))
            upper_bound = min(min(r[j], d[i] - row_sum), gale_ryser_condition)
            # Choose a value between bounds
            for s in range(lower_bound, upper_bound + 1):
                updated_d = xsi_tilde(deg_seq=d.copy(), s=s, value=j + 1)
                total += binomial(r[j], s) * kappa(updated_d, a, i, j + 1, row_sum + s, submatrix_count_values)

            return total


def sample(d, a, i, j, row_sum, sample_matrix, num_of_matrices, submatrix_count_values):
    """
    Recursive function to generate a sample realizing the degree sequence d
    :param d: degree sequence
    :param a: maximum element of the initial degree sequence d^0
    :param i: row index from 0 to n-1
    :param j: column index from 0 to a-1
    :param row_sum: Sum of chosen elements up to j
    :param sample_matrix:
    :param num_of_matrices:
    :param submatrix_count_values:
    :return:
    """

    if np.sum(d) == 0:
        return
    else:

        if d[i] == row_sum:
            # Set d[i] to 0
            d[i] = 0
            sample(d, a, i + 1, 0, 0, sample_matrix, num_of_matrices, submatrix_count_values)

        else:
            conj_r = np.append(conjugate(d[i + 1:], a), 0)
            conj_z = np.append(conjugate(d[i + j + 1:], a), 0)

            # Construct the vector of counts, r, from conjugate partition of q
            r = np.asarray([conj_r[inx - 1] - conj_r[inx] for inx in np.arange(1, len(conj_r))] + [conj_r[-1]])

            # Determine lower bound for the entry in ith row and jth column
            lower_bound = max(0, d[i] - row_sum - conj_r[j + 1])
            # Determine upper bound for the entry in ith row and jth column
            gale_ryser_condition = np.sum(conj_z[0:j + 1]) - np.sum(d[i + 1:(i + 1) + j + 1]) + ((j + 1) * (j + 2))
            upper_bound = min(min(r[j], d[i] - row_sum), gale_ryser_condition)

            # Sample uniformly from the set {0,1,...,num_of_matrices-1}
            random_number = rn.randint(0, num_of_matrices-1)

            # Choose a value between bounds
            total = 0

            for s in range(lower_bound, upper_bound + 1):

                updated_d = xsi_tilde(deg_seq=d, s=s, value=j + 1)
                value = kappa(updated_d.copy(), a, i, j + 1, row_sum + s, submatrix_count_values)

                total += binomial(r[j], s) * value
                if total > random_number:
                    sample_matrix[i][j] = s
                    sample(updated_d, a, i, j + 1, row_sum + s, sample_matrix, value, submatrix_count_values)
                    return

            raise Warning("Algorithm must not reach this line! Check the error!")


def count(deg_seq, verbose=False):
    """
    Count the number of simple graphs realizing the degree sequence deg_seq
    :param deg_seq: Degree sequence
    :param verbose:
    :return result: Number of simple graphs with degree sequence deg_seq
    """

    # Copy the sequence
    d = deg_seq.copy()

    # Get initial time
    time_start = time.clock()

    # If the sequence is empty
    if len(d) == 0:
        if verbose is True:
            # Get the total computation time
            time_elapsed = (time.clock() - time_start)
            print "Total computation time : " + str(time_elapsed)
        return 1

    # if the sequence is not graphical
    if is_graphical(d) is False:
        if verbose is True:
            # Get the total computation time
            time_elapsed = (time.clock() - time_start)
            print "Total computation time : " + str(time_elapsed)
        return 0

    # Sort d in non-increasing order
    d = np.sort(d)[::-1]

    # Store subMatrix counts to avoid from computing them again
    submatrix_count_values = {}

    result = kappa(d, max(d), i=0, j=0, row_sum=0, submatrix_count_values=submatrix_count_values)

    # Get the total computation time
    time_elapsed = (time.clock() - time_start)
    if verbose is True:
        print "Total computation time : " + str(time_elapsed)

    return result


def get_sample(deg_seq, num_of_samples, verbose=False):
    """
    Generates uniformly distributed samples satisfying a given degree sequence
    :param deg_seq: Degree sequence
    :param num_of_samples: Number of samples which will be generated
    :param verbose:
    :return edges: Sequence of edge sequences, vertex labels start from 1
    """

    # Copy the sequence
    d = deg_seq.copy()

    # Get initial time
    time_start = time.clock()

    # If the sequence is empty or is not graphical
    if len(d) == 0 or is_graphical(d) is False:
        if verbose is True:
            # Get the total computation time
            time_elapsed = (time.clock() - time_start)
            print "Total computation time : " + str(time_elapsed)
        return []

    # Get the size of the sequence d
    n = len(d)
    # Get the maximum element of d
    a = np.max(d)

    # Store the sorted indices
    inx_order = np.argsort(d)
    inx_order = inx_order[::-1]
    inx_order = np.argsort(inx_order)
    # Sort d in descending order
    d = np.sort(d)[::-1]

    # Store subMatrix counts to avoid from computing them again
    submatrix_count_values = {}

    num_of_matrices = kappa(d.copy(), a, i=0, j=0, row_sum=0, submatrix_count_values=submatrix_count_values)

    edges = []
    for k in range(num_of_samples):
        sample_matrix_s = np.zeros((n, a), dtype=np.int)

        sample(d.copy(), max(d), 0, 0, 0, sample_matrix_s, num_of_matrices, submatrix_count_values)
        binary_matrix = convert2binary(d, sample_matrix_s)
        binary_matrix = binary_matrix[inx_order, :][:, inx_order]

        e = matrix2edges(binary_matrix, "simple")
        edges.append(e)

    # Get the total computation time
    time_elapsed = (time.clock() - time_start)
    if verbose is True:
        print "Total computation time : " + str(time_elapsed)

    return edges