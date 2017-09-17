from sacorg.utils import *
from sacorg.algorithms.generate_graph import *


def compute_perfect_matchings(vertices):
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
        if np.sum([p[2 * i + 2] > p[2 * i] for i in range(n / 2 - 1)]) == n / 2 - 1 and \
                        np.sum([p[2 * i + 1] > p[2 * i] for i in range(n / 2)]) == n / 2:
            # Permute the vertices
            permuted = [vertices[i] for i in p]
            # Append the permuted sequence
            matchings.append([[permuted[i * 2], permuted[i * 2 + 1]] for i in range(n / 2)])

    return matchings


def sample(initial_edges, iteration=-1):
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
        iteration = 100 * num_of_edges

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
            matchings = compute_perfect_matchings(vertices)
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


def get_sample(deg_seq, num_of_samples, iteration=-1, verbose=False):
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
        e, switching_count = sample(initial_edges=initial_e, iteration=iteration)
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