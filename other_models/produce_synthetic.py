import networkx as nx
import os

def generate_graphs():
    if not os.path.exists("synthetic"):
        os.makedirs("synthetic")
        
    print("Generating BA_1k_3...")
    # BA n=1000, m=3 (degree approx 6, but in networkx m is edges attached per new node)
    # The paper says average degree 3, so we can use m=2 or 3. Let's use 3.
    gba1 = nx.barabasi_albert_graph(1000, 3, seed=42)
    nx.write_edgelist(gba1, "synthetic/BA_1k_3.txt", data=False)

    print("Generating BA_1k_4...")
    gba2 = nx.barabasi_albert_graph(1000, 4, seed=43)
    nx.write_edgelist(gba2, "synthetic/BA_1k_4.txt", data=False)

    print("Generating ER_1k_3...")
    # ER n=1000, avg deg 3 -> p = 3/1000 = 0.003
    ger1 = nx.erdos_renyi_graph(1000, 0.003, seed=44)
    nx.write_edgelist(ger1, "synthetic/ER_1k_3.txt", data=False)

    print("Generating ER_1k_4...")
    ger2 = nx.erdos_renyi_graph(1000, 0.004, seed=45)
    nx.write_edgelist(ger2, "synthetic/ER_1k_4.txt", data=False)
    print("Done generating synthetic graphs.")

if __name__ == "__main__":
    generate_graphs()
