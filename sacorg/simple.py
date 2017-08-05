import time
import random
import numpy as np
from utils import binomial


def is_graphical(deg_seq, method="Erdos-Gallai", is_sorted=False):
    """
        Checks whether given degree sequence d is graphical or not
        :param deg_seq: given degree sequence
        :param method: method to check given degree sequence is graphical or not
        :param is_sorted
        :return: boolean value representing whether given sequence is graphical or not
    """
    # Copy the given degree sequence
    d = np.asarray(deg_seq).copy()


    # All degrees must be greater than zero
    #if np.any(d < 0):
    #    return False

    # The sum of degrees must be even
    if sum(d) % 2:
        return False

    # Sort the sequence in non-increasing order
    if is_sorted is False:
        d = np.sort(d)[::-1]

    # Get the length of the sequence
    N = len(d)

    """
        Implementation of Erdos-Gallai Theorem
    """
    if method == "Erdos-Gallai":

        # Check all n conditions
        for k in range(1, N + 1):
            if sum(d[0:k]) > k * (k - 1) + sum([min(d_i, k) for d_i in d[k:N]]):
                return False
        # Return true, if all conditions are satisfied
        return True


class MyAlg():
    """
    Exact counting and uniform sampling algorithm from the set of simple graphs realizing the given degree sequence
    """

    def __init__(self):
        pass

    def vector_of_counts(self, deg_seq, dim=-1):
        """
        Compute the vector of counts for deg_seq
        :param deg_seq: degree sequence
        :param dim: length of the output sequence
        :return vector_of_counts: vector of counts for deg_seq
        """
        if dim < 0:
            dim = max(deg_seq)

        vector_of_counts = np.asarray([np.sum(deg_seq == value) for value in range(1, dim+1)])
        return vector_of_counts

    def conjugate(self, deg_seq, dim=-1):
        """
        Computes the conjugate of the given degree sequence deg_seq
        :param deg_seq: degree sequence
        :param dim: length of the output sequence
        :return conj_d: conjugate of deg_seq
        """
        if dim < 0:
            dim = max(deg_seq)

        if len(deg_seq) > 0:
            conj_d = np.asarray([np.sum(deg_seq >= num) for num in np.arange(1, dim + 1)])
        else:
            conj_d = np.zeros((1, dim), dtype=np.int)

        return conj_d

    def xsi_tilde(self, deg_seq, s, value):
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
            if d[-1-inx] == value:
                d[-1-inx] -= 1
                count += 1
            inx += 1

        return d

    def convert2binary(self, deg_seq, matrix_s):
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

                for val in range(1, a+1):
                    col_indices = [inx for inx in range(n) if d[inx] == val and row != inx]
                    if len(col_indices) > 0:
                        chosen_indices = np.random.choice(col_indices, matrix_s[i][val-1], replace=False)

                        binary_matrix[row, chosen_indices] = 1
                        d[chosen_indices] -= 1

                d[row] = 0

        binary_matrix = binary_matrix + binary_matrix.T

        return binary_matrix

    def matrix2edges(self, matrix):
        """
        Converts a given binary matrix to the corresponding set of edges
        :param matrix: Symmetric binary matrix having zero diagonal entries
        :return edges: Set of edges
        """
        # Get the row and column size
        n = matrix.shape[0]
        # Ouput edge sequence
        edges = []
        for i in range(n):
            for j in range(i+1,n):
                if matrix[i, j] == 1:
                    edges.append([i+1,j+1])

        return edges

    def kappa(self, d, a, i, j, row_sum, submatrix_count_values):
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
                count_inx = tuple(self.vector_of_counts(d[i+1:], dim=a))
                if count_inx in submatrix_count_values:
                    return submatrix_count_values[count_inx]
                else:
                    total = self.kappa(d, a, i+1, 0, 0, submatrix_count_values)
                    submatrix_count_values[count_inx] = total
                    return total

            else:
                conj_r = np.append(self.conjugate(d[i+1:], a), 0)
                conj_z = np.append(self.conjugate(d[i+j+1:], a), 0)

                # Construct the vector of counts, r, from conjugate partition of q
                r = np.asarray([conj_r[inx - 1] - conj_r[inx] for inx in np.arange(1, len(conj_r))] + [conj_r[-1]])

                # Determine lower bound for the entry in ith row and jth column
                lower_bound = max(0, d[i] - row_sum - conj_r[j + 1])
                # Determine upper bound for the entry in ith row and jth column
                gale_ryser_condition = np.sum(conj_z[0:j + 1]) - np.sum(d[i + 1:(i + 1) + j + 1]) + ((j+1)*(j+2))
                upper_bound = min(min(r[j], d[i] - row_sum), gale_ryser_condition)
                # Choose a value between bounds
                for s in range(lower_bound, upper_bound + 1):

                    updated_d = self.xsi_tilde(deg_seq=d.copy(), s=s, value=j+1)
                    total += binomial(r[j], s) * self.kappa(updated_d, a, i, j+1, row_sum + s, submatrix_count_values)

                return total

    def sample(self, d, a, i, j, row_sum, sample_matrix, num_of_matrices, submatrix_count_values):
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
                self.sample(d, a, i+1, 0, 0, sample_matrix, num_of_matrices, submatrix_count_values)

            else:
                conj_r = np.append(self.conjugate(d[i + 1:], a), 0)
                conj_z = np.append(self.conjugate(d[i + j + 1:], a), 0)

                # Construct the vector of counts, r, from conjugate partition of q
                r = np.asarray([conj_r[inx - 1] - conj_r[inx] for inx in np.arange(1, len(conj_r))] + [conj_r[-1]])

                # Determine lower bound for the entry in ith row and jth column
                lower_bound = max(0, d[i] - row_sum - conj_r[j + 1])
                # Determine upper bound for the entry in ith row and jth column
                gale_ryser_condition = np.sum(conj_z[0:j + 1]) - np.sum(d[i + 1:(i + 1) + j + 1]) + ((j + 1) * (j + 2))
                upper_bound = min(min(r[j], d[i] - row_sum), gale_ryser_condition)

                # Sample uniformly from the set {0,1,...,num_of_matrices-1}
                random_number = random.randint(0, (num_of_matrices - 1))

                # Choose a value between bounds
                total = 0

                for s in range(lower_bound, upper_bound + 1):

                    updated_d = self.xsi_tilde(deg_seq=d, s=s, value=j + 1)
                    value = self.kappa(updated_d.copy(), a, i, j+1, row_sum + s, submatrix_count_values)

                    total += binomial(r[j], s) * value
                    if total > random_number:
                        sample_matrix[i][j] = s
                        self.sample(updated_d, a, i, j+1, row_sum + s, sample_matrix, value, submatrix_count_values)
                        return

                raise Warning("Algorithm must not reach this line! Check the error!")

    def count(self, deg_seq, verbose=False):
        """
        Count the number of simple graphs realizing the degree sequence deg_seq
        :param deg_seq: Degree sequence
        :param verbose:
        :return result: Number of simple graphs with degree sequence deg_seq
        """

        # Copy the sequence
        d = deg_seq.copy()

        # If the sequence is empty
        if len(d) == 0:
            return 1
        # if the sequence is not graphical
        if is_graphical(d) is False:
            return 0

        # Get initial time
        time_start = time.clock()

        # Sort d in non-increasing order
        d = np.sort(d)[::-1]

        # Store subMatrix counts to avoid from computing them again
        submatrix_count_values = {}

        result = self.kappa(d, max(d), i=0, j=0, row_sum=0, submatrix_count_values=submatrix_count_values)

        # Get the total computation time
        time_elapsed = (time.clock() - time_start)
        if verbose is True:
            print "Total computation time : " + str(time_elapsed)

        return result

    def get_sample(self, deg_seq, num_of_samples, verbose=False):
        """
        Uniformly generates simple graphs satisfying the given degree sequence
        :param deg_seq: Degree sequence
        :param num_of_samples: Number of samples which will be generated
        :return edges: Sequence of edge sequences
        """

        # Copy the sequence
        d = deg_seq.copy()

        # If the sequence is empty or is not graphical
        if len(d) == 0 or is_graphical(d) is False:
            return []

        # Get initial time
        time_start = time.clock()

        # Get the size of the sequence d
        n = len(d)
        # Get the maximum element of d
        a = np.max(d)

        # Store the sorted indices
        inx_order = np.argsort(d)
        inx_order = inx_order[::-1]
        inx_order = np.argsort(inx_order)
        # Sort d in ascending order
        d = np.sort(d)[::-1]

        # Store subMatrix counts to avoid from computing them again
        submatrix_count_values = {}

        num_of_matrices = self.kappa(d.copy(), a, i=0, j=0, row_sum=0, submatrix_count_values=submatrix_count_values)

        edges = []
        for k in range(num_of_samples):
            sample_matrix_s = np.zeros((n, a), dtype=np.int)

            self.sample(d.copy(), max(d), 0, 0, 0, sample_matrix_s, num_of_matrices, submatrix_count_values)
            binary_matrix = self.convert2binary(d, sample_matrix_s)
            binary_matrix = binary_matrix[inx_order, :][:, inx_order]

            e = self.matrix2edges(binary_matrix)
            edges.append(e)

        # Get the total computation time
        time_elapsed = (time.clock() - time_start)
        if verbose is True:
            print "Total computation time : " + str(time_elapsed)

        return edges
