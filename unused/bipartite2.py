import time

import numpy as np

from sacorg.utils.utils import binomial


def is_bigraphic(p, q, method="Gale-Ryser"):
    """
        Checks whether given degree sequences of partite sets (p,q) is bigraphic or not
        :param p: degree sequence of a parite set
        :param q: degree sequence of the other parite set
        :param method: method to check given degree sequence is graphical or not
        :param is_sorted
        :return: boolean value representing whether given sequence is graphical or not
    """
    # Copy the given degree sequences
    d1 = np.asarray(p).copy()
    d2 = np.asarray(q).copy()

    # If the length of the sequences is 0, it is bigraphic
    if len(d1) == 0 and len(d2) == 0:
        return True

    # All degrees must be greater than zero
    if np.any(d1 < 0) or np.any(d2 < 0):
        return False

    # The sum of degree sequences must be equal
    if sum(d1) != sum(d2):
        return False

    # Maximum of a sequence cannot be larger than the number of non-zero elements in the other sequence
    if np.sum(d1 > 0) < max(d2) or np.sum(d2 > 0) < max(d1):
        return False

    """
        Implementation of Gale-Ryser Theorem
    """
    if method == "Gale-Ryser":

        # Consider only the elements greater than zero in the sequences
        d1 = np.asarray([x for x in d1 if x > 0])
        d2 = np.asarray([x for x in d2 if x > 0])

        # Get the length of the sequence d1
        N1 = len(d1)

        # Find the conjugate of the sequence d2 with the size of N1
        conj_d2 = [np.sum(d2 >= x) for x in range(1, N1+1)]

        # Checkthe conditions
        for k in range(1, N1+1):
            if sum(d1[0:k]) > sum(conj_d2[0:k]):
                return False
        # Return true, if all conditions are satisfied
        return True


class ChenDiaconisHolmesLiu:
    """
    The Weighted Sampling Algorithm for Generating Bipartite Graphs or Binary Matrices Satisfying Given Margins

    "Sequential Monte Carlo Methods for Statistical Analysis of Tables"
    Yuguo CHEN, Persi DIACONIS, Susan P. HOLMES, and Jun S. LIU
    """

    def __init__(self):
        pass

    def r(self, w, s):
        """
        The R(s,A) function
        :param w: Weightes
        :param s: Suibset size
        :return: Sum of product of all size 's' subsets of the set set 'w'
        """
        # Here, s must be at most numOfRows - 1
        num_of_rows = len(w)

        r = np.zeros(num_of_rows)

        # t is a sequence containing the sums of powers of weights, ie 0, a+b+c, (a+b+c)^2
        t = np.asarray([0.0] + [float(np.sum(w[0:] ** i)) for i in range(1, s+1)])

        r[0] = 1
        for i in range(1, s + 1):
            r[i] = 0
            for j in range(1, i + 1):
                if j % 2 == 0:
                    r[i] = r[i] - t[j] * r[i - j]
                else:
                    r[i] = r[i] + t[j] * r[i - j]
            r[i] = r[i] / float(i)

        return r[s]

    def drafting_sampling(self, c, p):
        """
        Drafting sampling algorithm
        :param c: The number of elements which will be picked from {0,...,len(p)}
        :param p: The probabilities
        :return: The chosen elements
        """

        # Copy the probabilities
        copy_p = p.copy()

        # Determine the set size which will be used to sample 'c' elements from
        n = len(p)

        # Get the indices whose probabilities equal to 1
        positions = list(np.where(copy_p == 1)[0])

        # If there are more than 'c' indices having probability 1, choose c elements among them
        if len(positions) >= c:
            return np.random.choice(positions, size=c, replace=False)
        # Otherwise, choose all elements having probability 1
        else:
            chosen_elements = positions
            # Set the probabilities of the chosen elements to 0
            copy_p[chosen_elements] = 0

        # Compute the weights
        w = np.divide(copy_p, (1-copy_p))
        # Copy the weights
        current_weights = w.copy()

        k = len(chosen_elements) + 1
        while k <= c:
            positions = np.where(current_weights > 0)[0]
            if c-k+1 == len(positions):
                for inx in positions:
                    chosen_elements.append(inx)
                break

            prob = np.zeros(n, dtype=np.float)
            for i in range(n):
                if current_weights[i] > 0:
                    temp_weight = current_weights[i]
                    current_weights[i] = 0
                    prob[i] = temp_weight * self.r(w=current_weights, s=c-k)
                    current_weights[i] = temp_weight

            prob = prob / np.sum(prob)
            inx = np.random.choice(positions, p=prob[positions], size=1)[0]

            chosen_elements.append(inx)
            current_weights[inx] = 0

            k += 1

        return chosen_elements

    def generate_table(self, p, q):
        """
        Generaye a binary matrix satisfying the row sum p and column sum q
        :param p: The row sum sequence
        :param q: The column sum sequence
        :return binary_matrix, valid_sample: A binary matrix satisfying given margins if 'valid_sanple' is true
        """

        # Output matrix
        binary_matrix = []
        # Check the generated matrix is valid or not
        valid_sample = True
        # Get the row and column size
        n = len(p)
        m = len(q)
        # Copy the row sum
        row_sum = p.copy()

        for j in range(m):
            # If column sum is zero, all entries must be 0
            if q[j] == 0:
                col = [0 for _ in range(n)]
            # If column sum is equal to length of the column, all entries must be 1
            elif q[j] == n:
                col = [1 for _ in range(n)]
            # Check the updated margin sums is bigraphic, otherwise the sample is invalid
            elif is_bigraphic(p=row_sum, q=q[j:], method="Gale-Ryser") is False:
                valid_sample = False
                break
            # Choose q[j] positions to place 1's
            else:
                chosen_indices = self.drafting_sampling(c=q[j], p=row_sum/float(m-j))
                col = [1 if inx in chosen_indices else 0 for inx in range(n)]

            binary_matrix.append(col)
            row_sum -= col

        return binary_matrix, valid_sample

    def get_sample(self, p, q, num_of_samples, verbose=False):
        """
        Generates uniformly distributed samples satisfying a given degree sequence
        :param p: Row sum sequence
        :param q: Column sum sequence
        :param num_of_samples: Number of samples which will be generated
        :param verbose:
        :return binary matrix: Binary matrices satisfying the given margins (p,q)
        """

        # Copy the margin sum
        d1, d2 = p.copy(), q.copy()

        # Get initial time
        time_start = time.clock()

        # If the sequence is empty or is not graphical
        if len(d1) == 0 or len(d2) == 0 or is_bigraphic(d1, d2) is False:
            if verbose is True:
                # Get the total computation time
                time_elapsed = (time.clock() - time_start)
                print "Total computation time : " + str(time_elapsed)
            return []

        # Store the sorted indices
        inx_order1, inx_order2 = np.argsort(d1), np.argsort(d2)
        inx_order1, inx_order2 = inx_order1[::-1], inx_order2[::-1]
        inx_order1, inx_order2 = np.argsort(inx_order1), np.argsort(inx_order2)

        # Sort sequences in descending order to decrease the number of invalid samples
        d1, d2 = np.sort(d1)[::-1], np.sort(d2)[::-1]

        matrices = []
        for k in range(num_of_samples):

            m, valid = self.generate_table(d1, d2)

            if valid is True:
                m = np.asarray(m).T
                m = m[inx_order1, :][:, inx_order2]
                matrices.append(m)

        # Get the total computation time
        time_elapsed = (time.clock() - time_start)
        if verbose is True:
            print "Total computation time : " + str(time_elapsed)
            print str(len(matrices)) + " valid samples were generated out of " + str(num_of_samples) + " trials."

        return matrices


class MillerHarrison:
    """
        Uniform Sampling and Exact Counting Algorithm for Generating Bipartite Graphs or Binary Matrices Satisfying Given Margins

        "Exact sampling and counting for fixed-margin matrices"
        J. W. Miller and M. T. Harrison
        """

    def __init__(self):
        pass

    def vector_of_counts(self, deg_seq):
        """
        Compute the vector of counts for deg_seq
        :param deg_seq: degree sequence
        :return vector_of_counts: vector of counts for deg_seq
        """
        vector_of_counts = np.asarray([np.sum(deg_seq == value) for value in range(1, max(deg_seq)+1)])
        return vector_of_counts

    def conjugate(self, deg_seq):
        """
        Computes the conjugate of the given degree sequence deg_seq
        :param deg_seq: degree sequence
        :return conj_d: conjugate of deg_seq
        """

        conj_d = np.asarray([np.sum(deg_seq >= num) for num in np.arange(1, max(deg_seq) + 1)])
        return conj_d

    def N_count(self, p, conj_q, i, j, row_sum, submatrix_count_values):
        """
        Computes the number of binary matrices with margin sums (p,q)
        :param p: Row sum vector
        :param conj_q: Conjugate of the column vector q
        :param i: Row index
        :param j: Column index
        :param row_sum: Sum of the chosen row entries up to (i-1)th coordinate
        :param submatrix_count_values: Lookup table storing intermediate results
        :return total: The number of binary matrix with row and column sums p, q
        """
        total = 0
        if len(p) == i:
            return 1
        else:
            if p[i] == row_sum:
                # Store submatrix counts to avoid from computing them each time
                count_inx = tuple(conj_q)
                if count_inx in submatrix_count_values:
                    return submatrix_count_values[count_inx]

                total = self.N_count(p, conj_q, i+1, 0, 0, submatrix_count_values)
                submatrix_count_values[count_inx] = total

                return total

            else:
                # Construct the vector of counts, r, from conjugate partition of q
                r = np.asarray([conj_q[inx - 1] - conj_q[inx] for inx in np.arange(1, len(conj_q))] + [conj_q[-1]])

                # Determine lower bound for the entry in ith row and jth column
                lower_bound = max(0, p[i] - row_sum - conj_q[j + 1])
                # Determine upper bound for the entry in ith row and jth column
                gale_ryser_condition = np.sum(conj_q[0:j+1]) - np.sum(p[i+1:(i+1)+j+1])
                upper_bound = min(min(r[j], p[i] - row_sum), gale_ryser_condition)

                # Choose a value between bounds
                for s in range(lower_bound, upper_bound + 1):
                    conj_q[j] -= s
                    total += binomial(r[j], s) * self.N_count(p, conj_q, i, j+1, row_sum + s, submatrix_count_values)
                    conj_q[j] += s

                return total

    def N_sample(self, p, conj_q, i, j, row_sum, sample_matrix, num_of_matrices, submatrix_count_values):
        """
        Generates a uniformly distributed sample satisfying margins (p, q_vect_counts)
        :param p: Row sum vector
        :param conj_q: Column sum vector
        :param i: Row index
        :param j: Column index
        :param row_sum: Sum of the chosen row entries up to (i-1)th coordinate
        :param sample_matrix: Constructed sample matrix with row and column sums (p, q_vect_counts)
        :param num_of_matrices: The number of matrices for the remaining submatrix
        :param submatrix_count_values: Lookup table storing intermediate results
        """

        if len(p) == i:
            return
        else:
            if p[i] == row_sum:

                self.N_sample(p, conj_q.copy(), i+1, 0, 0, sample_matrix, num_of_matrices, submatrix_count_values)

            else:
                # Construct the vector of counts, r, from conjugate partition of q
                r = np.asarray([conj_q[inx - 1] - conj_q[inx] for inx in np.arange(1, len(conj_q))] + [conj_q[-1]])

                # Determine lower bound for the entry in ith row and jth column
                lower_bound = max(0, p[i] - row_sum - conj_q[j + 1])
                # Determine upper bound for the entry in ith row and jth column
                gale_ryser_condition = np.sum(conj_q[0:j+1]) - np.sum(p[i+1:(i+1)+j+1])
                upper_bound = min(min(r[j], p[i] - row_sum), gale_ryser_condition)
                # Sample uniformly from the set {0,1,...,num_of_matrices-1}
                random_number = np.random.randint(num_of_matrices, dtype=np.int64)

                # Choose a value between bounds
                total = 0
                for s in range(lower_bound, upper_bound + 1):
                    conj_q[j] -= s
                    value = self.N_count(p, conj_q.copy(), i, j+1, row_sum + s, submatrix_count_values)
                    total += binomial(r[j], s) * value

                    if total > random_number:
                        sample_matrix[i][j] = s
                        self.N_sample(p, conj_q.copy(), i, j+1, row_sum + s, sample_matrix, value, submatrix_count_values)
                        conj_q[j] += s
                        return

                    conj_q[j] += s

                raise Warning("Algorithm must not reach this line! Check the error!")

    def convert2binary(self, n, m, q, matrix_s):
        """
        Converts the matrix generated by N_sample function to a binary matrix
        :param n: Number of rows
        :param m: Number of columns
        :param q: Column sum
        :param matrix_s:
        :return binary_matrix: A binary matrix corresponding to N-valued (p, q_vect_counts)
        """

        # Copy the column sum q
        column_sum = np.asarray(q).copy()

        binary_matrix = np.zeros((n, m), dtype=np.int)
        for i in range(n):
            for j in range(max(q)):
                if matrix_s[i][j] > 0:
                    indices = np.where(column_sum == j+1)[0]
                    # Randomly choose indices
                    choosen_indices = np.random.choice(indices, matrix_s[i][j], replace=False)
                    column_sum[choosen_indices] -= 1
                    binary_matrix[i][choosen_indices] = 1

        return binary_matrix

    def count(self, p, q, verbose=False):
        """
        Counts the number of binary matrices satisfying the margin sums (p,q)
        :param p: Row sum sequence
        :param q: Column sum sequence
        :return result: Number of binary matrices satisfying given margins
        """

        # Copy the sequences
        p_copy, q_copy = p.copy(), q.copy()

        # Get initial time
        time_start = time.clock()

        # If the sequences are empty, return 1
        if len(p_copy) == 0 and len(q_copy) == 0:
            if verbose is True:
                # Get the total computation time
                time_elapsed = (time.clock() - time_start)
                print "Total computation time : " + str(time_elapsed)
            return 1

        # If given margin sums are not bigraphic
        if is_bigraphic(p_copy, q_copy) is False:
            if verbose is True:
                # Get the total computation time
                time_elapsed = (time.clock() - time_start)
                print "Total computation time : " + str(time_elapsed)
            return 0

        # If the length of the sequence p is larger than q, than switch the sequences
        if len(q_copy) > len(p_copy):
            temp = p_copy
            p_copy = q_copy
            q_copy = temp

        # Find the conjugate of q and append 0 to avoid from index exceeding problem
        q_conj = self.conjugate(q_copy)
        q_conj = np.append(q_conj, 0)

        # Sort p in descending order
        p_copy = np.sort(p_copy)[::-1]

        # Store subMatrix counts to avoid from computing them again
        submatrix_count_values = {}

        result = self.N_count(p_copy, q_conj.copy(), i=0, j=0, row_sum=0, submatrix_count_values=submatrix_count_values)
        return result

    def get_sample(self, p, q, num_of_samples, verbose=False):
        """
        Uniformly generates a binary matrix satisfying the row and column sums (p,q)
        :param p: Row sum sequence
        :param q: Column sum sequence
        :param num_of_samples: Number of samples that will be generated
        :return samples: Binary samples
        """

        # Copy the sequences
        p_copy, q_copy = p.copy(), q.copy()

        # Get initial time
        time_start = time.clock()

        # If the sequence is empty or is not graphical
        if (len(p_copy) == 0 and len(q_copy) == 0) or is_bigraphic(p_copy, q_copy) is False:
            if verbose is True:
                # Get the total computation time
                time_elapsed = (time.clock() - time_start)
                print "Total computation time : " + str(time_elapsed)
            return []

        # The length of p should be larger than q, otherwise switch p and q
        transposed = False
        if len(q_copy) > len(p_copy):
            temp = p_copy
            p_copy = q_copy
            q_copy = temp
            transposed = True

        # Store the sorted indices
        inx_order1, inx_order2 = np.argsort(p_copy), np.argsort(q_copy)
        inx_order1, inx_order2 = inx_order1[::-1], inx_order2[::-1]
        inx_order1, inx_order2 = np.argsort(inx_order1), np.argsort(inx_order2)
        # Sort sequences in descending order to increase the performance
        p_copy, q_copy = np.sort(p_copy)[::-1], np.sort(q_copy)[::-1]

        # Find the conjugate of q and append 0 to avoid from index exceeding problem
        q_conj = self.conjugate(q_copy)
        q_conj = np.append(q_conj, 0)
        # Store subMatrix counts to avoid from computing them again
        submatrix_count_values = {}
        # Compute the number of matrices satisfying the margins p,q
        num_of_matrices = self.N_count(p_copy, q_conj, i=0, j=0, row_sum=0, submatrix_count_values=submatrix_count_values)

        matrices = []
        for k in range(num_of_samples):

            sample_matrix_s = np.zeros((len(p_copy), max(q_copy)), dtype=np.int)
            self.N_sample(p_copy, q_conj, 0, 0, 0, sample_matrix_s, num_of_matrices, submatrix_count_values)

            m = self.convert2binary(len(p_copy), len(q_copy), q_copy, sample_matrix_s)
            m = m[inx_order1, :][:, inx_order2]
            if transposed is True:
                m = np.asarray(m).T

            matrices.append(m)

        # Get the total computation time
        time_elapsed = (time.clock() - time_start)
        if verbose is True:
            print "Total computation time : " + str(time_elapsed)
            print str(len(matrices)) + " valid samples were generated out of " + str(num_of_samples) + " trials."

        return matrices
