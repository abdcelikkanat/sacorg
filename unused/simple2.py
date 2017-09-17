import random
import time
from itertools import permutations as perm
from math import factorial

import numpy as np

from sacorg.utils.utils import binomial


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

    # If the length of the sequence is 0, it is graphical
    if len(d) == 0:
        return True

    # All degrees must be greater than zero
    if np.any(d < 0):
        return False

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


def generate_graph(deg_seq, method="Havel-Hakimi"):
    """
    Generates a graph satisfying a given degree sequence by using the indicated method
    :param deg_seq: Degree sequence
    :param method: The method which will be used to generate graph
    :return: A graph satisfying the degree sequence deg_seq with vertices starting from 1
    """

    # Copy the degree sequence
    res_seq = deg_seq.copy()

    edges = []
    # If the length of the sequence is zero or all elements are 0
    if len(res_seq) == 0 or np.all(res_seq == 0):
        return edges

    if method == "Havel-Hakimi":

        # Continue until all elements of the degree sequence become 0
        while np.any(res_seq > 0):

            # Sort the sequence in descending order
            sorted_inx = np.argsort(res_seq)[::-1]
            # Choose a vertex having non-zero degree
            chosen_inx = np.random.choice(np.where(res_seq > 0)[0], size=1)[0]

            i = 0
            while res_seq[chosen_inx] > 0:
                if sorted_inx[i] != chosen_inx:
                    # Add edges where each pair of vertices is placed in increasing order
                    if sorted_inx[i] < chosen_inx:
                        edges.append([sorted_inx[i]+1, chosen_inx+1])
                    else:
                        edges.append([chosen_inx+1, sorted_inx[i]+1])

                    # Subtract by 1
                    res_seq[chosen_inx] -= 1
                    res_seq[sorted_inx[i]] -= 1
                i += 1

        return edges


class MCMC:
    """
    Edge-switching based MCMC method to generate uniformly distributed samples
    from the set of simple graphs realizing a given degree sequence
    """

    def __init__(self):
        pass

    def compute_perfect_matchings(self, vertices):
        """
        Computes all possible perfect matchings of a given vertex set
        :param vertices: A set of vertices
        :return matchings: All perfect matchings of the vertex set 'vertices'
        """
        # All elements must be different from each other
        assert len(set(vertices)) == len(vertices), "All elements must be unique"
        # The number of elements in vertices must be even
        assert len(vertices) % 2 == 0, "The number of elements must be even"

        # Get the number of vertices
        n = len(vertices)

        matchings = []
        # Choose 2-cycles in the form of (01)...(45)(67)
        inx = np.arange(0, n)
        for p in perm(inx):
            if np.sum([p[2*i+2] > p[2*i] for i in range(n/2-1)]) == n/2-1 and \
                            np.sum([p[2*i+1] > p[2*i] for i in range(n / 2)]) == n/2:
                # Permute the vertices
                permuted = [vertices[i] for i in p]
                # Append the permuted sequence
                matchings.append([[permuted[i*2], permuted[i*2+1]] for i in range(n/2)])

        return matchings

    def sample(self, initial_edges, iteration=-1):
        """
        Performs edge-swithings on the given edge set 'initial_edges'
        :param initial_edges: The initial edge set
        :param iteration: The number of iterations
        :return: Generates uniformly distributed sample
        """

        # Copy the edge set
        edges = list(initial_edges)

        # Get the number of edges
        num_of_edges = len(edges)

        # If the number of iterations is not stated
        if iteration < 0:
            # it has been shown that for many networks, iterations = 100m seem to be adequate by the article
            # "On the uniform generation of random graphs with prescribed degree sequences"
            # R. Milo, N. Kashtan, S. Itzkovitz, M. E. J. Newman, U. Alon
            iteration = 100*num_of_edges

        switching_count = 0
        for _ in range(iteration):

            # Uniformly choose a number from (0,1) at random
            r = np.random.uniform(low=0, high=1, size=1)
            # If r is greater than or equal to 1/2
            if r >= 0.5:
                # Choose two non-adjacent edges
                vertices = []
                chosen_edges_inx = []
                while len(vertices) != 4:
                    chosen_edges_inx = np.random.choice(range(num_of_edges), size=2, replace=False)
                    vertices = list(set(edges[chosen_edges_inx[0]] + edges[chosen_edges_inx[1]]))

                # Compute all possible matchings
                matchings = self.compute_perfect_matchings(vertices)
                # Uniformly choose one of them at random
                inx = np.random.choice(range(len(matchings)), size=1, replace=False)[0]

                # If the proposed edges are not in the edge set, perform switching
                chosen_matching = matchings[inx]
                check_edge1 = np.sum([chosen_matching[0] == e for e in edges])
                check_edge2 = np.sum([chosen_matching[1] == e for e in edges])

                if check_edge1 == 0 and check_edge2 == 0:
                    # Perform switching
                    edge1 = edges[chosen_edges_inx[0]]
                    edge2 = edges[chosen_edges_inx[1]]
                    edges.remove(edge1)
                    edges.remove(edge2)
                    edges.append(chosen_matching[0])
                    edges.append(chosen_matching[1])
                    switching_count += 1

        # Sort the edge sequences
        edges.sort()
        return edges, switching_count

    def get_sample(self, deg_seq, num_of_samples, iteration=-1, verbose=False):
        """
        Generates uniformly distributed samples satisfying a given degree sequence
        :param deg_seq:  Degree sequence
        :param num_of_samples: Number of samples which will be generated
        :param iteration: Number of iterations to generate each sample
        :param verbose:
        :return edges: Sequence of edge sequences, vertex labels start from 1
        """

        # Get the initial time
        time_start = time.clock()

        average_switching_count = 0.0
        # Generate an initial graph
        initial_e = generate_graph(deg_seq=deg_seq, method="Havel-Hakimi")
        edges = []
        for _ in range(num_of_samples):
            # Call the sample function
            e, switching_count = self.sample(initial_edges=initial_e,  iteration=iteration)
            # Append the output sample
            edges.append(e)
            # Count the total edge switchings
            average_switching_count += switching_count

            # Average edge swithings
            average_switching_count /= num_of_samples

        # Get the total computation time
        time_elapsed = (time.clock() - time_start)

        if verbose is True:
            print "Total computation time : " + str(time_elapsed)
            print "Average edge switching count : " + str(average_switching_count)

        # Return sequence of edge sequences
        return edges


class BlitzsteinDiaconis:
    """
    Implementation of sequential algorithm proposed by Blitzstein and Diaconis

    "A Sequential Importance Sampling Algorithm for Generating Random Graphs with Prescribed Degrees"
    Joseph Blitzstein and Persi Diaconis
    """
    def __init__(self):
        pass

    def s(self, deg_seq):
        """
        Generates a sample graph for a given degree sequence d
        :param deg_seq: Given degree sequence
        :return E, p, c:  edges with vertex labels starting from 1,
                          probability of the generated graph and
                          the number of edge combinations
        """
        # Copy the given degree sequence to use it as residual sequence
        r = deg_seq.copy()

        p = 1.0  # probability of the generated graph
        c = 1  # the number of edge combinations for the same graph that can be generated by the algorithm

        E = []  # list of edges
        N = len(r)  # length of the sequence

        adjacentVertices = [[] for _ in range(N)]  # stores the vertices which are adjacent

        # run until residual sequence completely becomes 0 vector
        while np.any(r != 0):

            # Get the index of vertex having minimum degree greater than 0
            i = np.where(r == np.amin(r[r > 0]))[0][-1]

            c *= factorial(r[i])

            while r[i] != 0:
                J = np.asarray([], dtype=np.int)  # Construct candidate list J

                possibleVertices = [o for o in np.arange(N) if (r[o] > 0 and o != i and (o not in adjacentVertices[i]))]
                for j in possibleVertices:
                    # Decrease degrees by one
                    (r[i], r[j]) = (r[i] - 1, r[j] - 1)
                    # add the the vertex j to candidate list J, if residual sequence is graphical
                    if is_graphical(r):
                        J = np.append(J, j)
                    # Increase degrees by one
                    (r[i], r[j]) = (r[i] + 1, r[j] + 1)

                # Pick a vertex j in the candidate list J with probability proportional to its degree d_j
                degrees = np.asarray([r[u] for u in J])
                prob = degrees / float(np.sum(degrees))
                j = np.random.choice(J, p=prob, size=1)[0]
                # Add the found edge to the edge lists
                if i < j:
                    E.append([i+1, j+1])  # indices start from 1
                else:
                    E.append([j+1, i+1])  # indices start from 1
                # Add the chosen vertex to the list in order to not choose it again
                adjacentVertices[i].append(j)
                # Decrease degrees by 1
                (r[i], r[j]) = (r[i] - 1, r[j] - 1)
                p *= prob[J == j][0]

        # Sort the edge sequences
        E.sort()
        return E, p, c

    def get_sample(self, deg_seq, num_of_samples, verbose=False):
        """
        Generates graphs realizing the degree sequence 'deg_seq' with vertex labels {1,...,len(deg_seq}}
        :param deg_seq: Degree sequence
        :param num_of_samples: Number of samples which will be generated
        :return:
        """
        # Get the initial time
        time_start = time.clock()

        # If the sequence is empty or is not graphical
        if len(deg_seq) == 0 or is_graphical(deg_seq) is False:
            if verbose is True:
                # Get the total computation time
                time_elapsed = (time.clock() - time_start)
                print "Total computation time : " + str(time_elapsed)
            return []

        edges = []
        for _ in range(num_of_samples):
            # Call the s function
            e, p, c = self.s(deg_seq)
            # Append the edges
            edges.append(e)

        # Get the total computation time
        time_elapsed = (time.clock() - time_start)

        if verbose is True:
            print "Total computation time : " + str(time_elapsed)

        # Return the edges
        return edges

    def count(self, deg_seq, num_of_samples=1000, verbose=False):
        """
        Estimates the number of graphs satisfying the degree sequence
        :param deq_seq: Degree sequence
        :param num_of_samples: number of samples used in estimation
        :return estimation, std: Estimation for the number of graphs satisfying the given degree sequence d
                                 and standard deviation
        """
        estimate = 0.0

        # Get initial time
        time_start = time.clock()

        # If the sequence is empty or is not graphical
        if len(deg_seq) == 0 or is_graphical(deg_seq) is False:
            if verbose is True:
                # Get the total computation time
                time_elapsed = (time.clock() - time_start)
                print "Total computation time : " + str(time_elapsed)
            return 0.0, 0.0

        weights = np.zeros(num_of_samples, dtype=float)

        for i in range(num_of_samples):
            (edges, p, c) = self.s(deg_seq)
            weights[i] = 1.0 / float(c * p)

        estimate = (1.0 / float(num_of_samples)) * np.sum(weights)
        std = np.std(weights, ddof=1)

        # Get the total computation time
        time_elapsed = (time.clock() - time_start)
        if verbose is True:
            print "Total computation time : " + str(time_elapsed)

        return estimate, std


class MyAlg:
    """
    Exact counting and uniform sampling algorithm from the set of simple graphs realizing a given degree sequence
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
        :return edges: Set of edges where vertex labels start from 1
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

        result = self.kappa(d, max(d), i=0, j=0, row_sum=0, submatrix_count_values=submatrix_count_values)

        # Get the total computation time
        time_elapsed = (time.clock() - time_start)
        if verbose is True:
            print "Total computation time : " + str(time_elapsed)

        return result

    def get_sample(self, deg_seq, num_of_samples, verbose=False):
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
