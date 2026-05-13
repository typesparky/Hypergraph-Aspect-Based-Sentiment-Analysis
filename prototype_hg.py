import spacy
import networkx as nx
import numpy as np

nlp = spacy.load("en_core_web_sm")

def construct_hg_and_spd(sentence):
    doc = nlp(sentence)
    # Build hypergraph as a bipartite graph: nodes + hyperedges
    G = nx.Graph()
    for token in doc:
        G.add_node(f"n{token.i}", type="node")
        for child in token.children:
            he_name = f"he{token.i}"
            G.add_node(he_name, type="hyperedge")
            G.add_edge(f"n{token.i}", he_name)
            G.add_edge(f"n{child.i}", he_name)
            
    # SPD matrix (only for token nodes)
    node_indices = [i for i in range(len(doc))]
    spd_matrix = np.zeros((len(doc), len(doc)), dtype=int)
    
    for i in node_indices:
        for j in node_indices:
            # Distance in bipartite graph / 2 to get node-to-node distance
            spd_matrix[i, j] = nx.shortest_path_length(G, f"n{i}", f"n{j}") // 2
            
    return spd_matrix

# Test
s = "The food is great."
print("SPD Matrix:\n", construct_hg_and_spd(s))
