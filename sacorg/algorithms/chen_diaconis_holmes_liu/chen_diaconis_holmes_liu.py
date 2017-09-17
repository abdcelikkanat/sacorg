from sacorg.utils import *
from sacorg.algorithms.isbigraphic import *


def r(w, s):
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
    t = np.asarray([0.0] + [float(np.sum(w[0:] ** i)) for i in range(1, s + 1)])

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


def drafting_sampling(c, p):
    """
    Drafting sampling algorithm
    :param c: The number of elements which will be picked from {0,...,len(p)}
    :param p: The probabilities
    :return: The chosen elements
    """

    # Copy the probabilities
    copy_p = p.copy()

    # Probability of chosen elements as output
    output_prob = 1.0

    # Determine the set size which will be used to sample 'c' elements from
    n = len(p)

    # Get the indices whose probabilities equal to 1
    positions = list(np.where(copy_p == 1)[0])

    # If there are more than 'c' indices having probability 1, choose c elements among them
    if len(positions) >= c:
        output_prob = float(c) / float(len(positions))
        return output_prob, np.random.choice(positions, size=c, replace=False)
    # Otherwise, choose all elements having probability 1
    else:
        chosen_elements = positions
        # Set the probabilities of the chosen elements to 0
        copy_p[chosen_elements] = 0

    # Compute the weights
    w = np.divide(copy_p, (1 - copy_p))
    # Copy the weights
    current_weights = w.copy()

    k = len(chosen_elements) + 1
    initial_chosen_elements_count = len(chosen_elements)
    while k <= c:
        positions = np.where(current_weights > 0)[0]
        if c - k + 1 == len(positions):
            for inx in positions:
                chosen_elements.append(inx)
            break

        prob = np.zeros(n, dtype=np.float)
        for i in range(n):
            if current_weights[i] > 0:
                temp_weight = current_weights[i]
                current_weights[i] = 0
                prob[i] = temp_weight * r(w=current_weights, s=c - k)
                current_weights[i] = temp_weight

        prob = prob / np.sum(prob)
        inx = np.random.choice(positions, p=prob[positions], size=1)[0]

        chosen_elements.append(inx)
        current_weights[inx] = 0
        output_prob *= prob[inx]

        k += 1

    output_prob *= factorial(c-initial_chosen_elements_count)

    return output_prob, chosen_elements


def generate_table(p, q):
    """
    Generaye a binary matrix satisfying the row sum p and column sum q
    :param p: The row sum sequence
    :param q: The column sum sequence
    :return binary_matrix, valid_sample: A binary matrix satisfying given margins if 'valid_sanple' is true
    """

    # Output matrix
    binary_matrix = []
    # Probability of generating the output matrix
    output_prob = 1.0
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
            output_prob = 0.0
            return [], output_prob, valid_sample
        # Choose q[j] positions to place 1's
        else:
            prob, chosen_indices = drafting_sampling(c=q[j], p=row_sum / float(m - j))
            col = [1 if inx in chosen_indices else 0 for inx in range(n)]
            output_prob *= prob

        binary_matrix.append(col)
        row_sum -= col

    return binary_matrix, output_prob, valid_sample


def get_sample(p, q, num_of_samples, verbose=False):
    """
    Generates uniformly distributed samples satisfying a given degree sequence
    :param p: Row sum sequence
    :param q: Column sum sequence
    :param num_of_samples: Number of samples which will be generated
    :param verbose:
    :return binary matrix: Binary matrices satisfying the given margins (p,q)
    """

    # Copy the margin sum
    p_copy, q_copy = p.copy(), q.copy()

    # Get initial time
    time_start = time.clock()

    # If the sequence is empty or is not graphical
    if len(p_copy) == 0 or len(q_copy) == 0 or is_bigraphic(p_copy, q_copy) is False:
        if verbose is True:
            # Get the total computation time
            time_elapsed = (time.clock() - time_start)
            print "Total computation time : " + str(time_elapsed)
        return []

    # Store the sorted indices
    inx_order1, inx_order2 = np.argsort(p_copy), np.argsort(q_copy)
    inx_order1, inx_order2 = inx_order1[::-1], inx_order2[::-1]
    inx_order1, inx_order2 = np.argsort(inx_order1), np.argsort(inx_order2)

    # Sort sequences in descending order to decrease the number of invalid samples
    p_copy, q_copy = np.sort(p_copy)[::-1], np.sort(q_copy)[::-1]

    #matrices = []
    edges = []
    for k in range(num_of_samples):

        m, prob, valid = generate_table(p_copy, q_copy)

        if valid is True:
            m = np.asarray(m).T
            m = m[inx_order1, :][:, inx_order2]
            #matrices.append(m)
            e = matrix2edges(m, "bipartite")
            edges.append(e)

    # Get the total computation time
    time_elapsed = (time.clock() - time_start)
    if verbose is True:
        print "Total computation time : " + str(time_elapsed)
        print str(len(edges)) + " valid samples were generated out of " + str(num_of_samples) + " trials."

    #return matrices
    return edges


def count(p, q, num_of_samples=1000, verbose=False):
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

    weights = []
    for k in range(num_of_samples):

        m, prob, valid = generate_table(p_copy, q_copy)

        if valid is True:
            weights.append(1.0 / prob)

    total = np.mean(weights)
    std = np.std(weights, ddof=1)

    # Get the total computation time
    time_elapsed = (time.clock() - time_start)
    if verbose is True:
        print "Total computation time : " + str(time_elapsed)
        print str(len(weights)) + " valid samples were generated out of " + str(num_of_samples) + " trials."

    return total, std