from sacorg.utils import *


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
