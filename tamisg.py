"""
TAMISG: A Metaheuristic Algorithm to Face the Graph Coloring Problem
=====================================================================
Reference: Guzmán-Ponce et al. (2020). MICAI 2020, Chapter 17,
           Springer LNAI 12469, pp. 196–208.

This demo implements the TAMISG algorithm which combines:
  1. GetMIS  — Deterministic construction of Maximal Independent Sets (MIS)
               from non-articulation vertices (Algorithm 2 in the paper).
  2. Tabu Search — Modified version of Tabucol to reduce coloring conflicts
                   while handling conflict paths (Algorithm 1 in the paper).
  3. E2COL logic — Extraction-and-Expansion method for graph coloring.

The chromatic number k is approximated: find the minimum k s.t. the graph
can be legally colored (no two adjacent vertices share a color).
"""

import random
import math
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Graph representation helpers
# ─────────────────────────────────────────────────────────────────────────────
class Graph:
    def __init__(self, vertices, edges):
        self.V = set(vertices)
        self.adj = defaultdict(set)
        for u, v in edges:
            self.adj[u].add(v)
            self.adj[v].add(u)
        self.E = edges

    def degree(self, v):
        return len(self.adj[v])

    def neighbors(self, v):
        return self.adj[v]

    def is_adjacent(self, u, v):
        return v in self.adj[u]

    def is_articulation(self, v):
        """Check if v is an articulation (cut) vertex via DFS."""
        remaining = self.V - {v}
        if not remaining:
            return False
        start = next(iter(remaining))
        visited = set()

        def dfs(u):
            visited.add(u)
            for w in self.adj[u]:
                if w in remaining and w not in visited:
                    dfs(w)

        dfs(start)
        return len(visited) < len(remaining)

    def is_bipartite(self):
        color = {}
        for start in self.V:
            if start in color:
                continue
            queue = [start]
            color[start] = 0
            while queue:
                u = queue.pop()
                for w in self.adj[u]:
                    if w not in color:
                        color[w] = 1 - color[u]
                        queue.append(w)
                    elif color[w] == color[u]:
                        return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Fitness function (Eq. 1 in the paper): count conflicting edges per coloring
# ─────────────────────────────────────────────────────────────────────────────
def fitness(coloring, graph, k=None):
    conflicts = 0
    for u, v in graph.E:
        if coloring.get(u) == coloring.get(v):
            conflicts += 1
    return conflicts


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 1 — Modified Tabu Search (Tabucol variant from TAMISG)
# ─────────────────────────────────────────────────────────────────────────────
def tabu_search(graph, k, max_iter=None, alpha=0.6):

    vertices = list(graph.V)
    if max_iter is None:
        max_iter = len(graph.E)          
    # Random initial coloring
    coloring = {v: random.randint(0, k - 1) for v in vertices}
    best_coloring = coloring.copy()
    best_f = fitness(coloring, graph, k)

    tabu = defaultdict(int)             # tabu[v] = iteration until which v is tabu

    for it in range(1, max_iter + 1):
        f_cur = fitness(coloring, graph, k)
        if f_cur == 0:
            return coloring, 0

        # Collect conflict vertices
        conflict_v = [v for v in vertices
                      if any(coloring[v] == coloring[u] for u in graph.neighbors(v))]
        if not conflict_v:
            break

        # Try all (vertex, new_color) moves; pick best non-tabu
        best_move_f = float("inf")
        best_move = None
        for v in conflict_v:
            if tabu[v] > it:
                continue
            for c in range(k):
                if c == coloring[v]:
                    continue
                old_c = coloring[v]
                coloring[v] = c
                new_f = fitness(coloring, graph, k)
                if new_f < best_move_f:
                    best_move_f = new_f
                    best_move = (v, c, old_c)
                coloring[v] = old_c

        if best_move is None:
            # All moves are tabu — aspiration: pick globally best
            for v in conflict_v:
                for c in range(k):
                    if c == coloring[v]:
                        continue
                    old_c = coloring[v]
                    coloring[v] = c
                    new_f = fitness(coloring, graph, k)
                    if new_f < best_move_f:
                        best_move_f = new_f
                        best_move = (v, c, old_c)
                    coloring[v] = old_c

        if best_move is None:
            break

        v, new_c, old_c = best_move
        coloring[v] = new_c
        tl = random.randint(0, 9) + int(alpha * best_move_f)
        tabu[v] = it + tl

        if best_move_f < best_f:
            best_f = best_move_f
            best_coloring = coloring.copy()

    return best_coloring, best_f


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 2 — GetMIS: build Maximal Independent Set from non-articulation vtx
# ─────────────────────────────────────────────────────────────────────────────
def get_mis(graph):

    all_v = list(graph.V)
    if not all_v:
        return set()

    # Pick starting non-articulation vertex with highest degree
    non_artic = [v for v in all_v if not graph.is_articulation(v)]
    if not non_artic:
        non_artic = all_v
    ve = max(non_artic, key=lambda v: graph.degree(v))

    K = {ve}
    # Candidates = non-neighbors of ve (sorted ascending by degree, per paper)
    no_neighbors = sorted(
        [u for u in all_v if u != ve and u not in graph.neighbors(ve)],
        key=lambda u: graph.degree(u)
    )

    excluded = set(graph.neighbors(ve)) | {ve}
    for candidate in no_neighbors:
        if candidate in excluded:
            continue
        if not graph.is_articulation(candidate):
            K.add(candidate)
            excluded |= graph.neighbors(candidate)
            excluded.add(candidate)

    return K


# ─────────────────────────────────────────────────────────────────────────────
# TAMISG main (E2COL-like extraction + MIS + modified Tabu Search)
# ─────────────────────────────────────────────────────────────────────────────
def tamisg(graph, k):
    """
    TAMISG: find a legal k-coloring (or best approximation).

    Returns
    -------
    coloring : dict {vertex: color}
    f_value  : int  number of conflicts (0 = legal coloring)
    """
    vertices = list(graph.V)
    n = len(vertices)
    # Residual graph keeps 40% of vertices (q = |V| - 60% of |V|)
    q = max(1, n - int(n * 0.60))

    # ── Extraction Phase ──────────────────────────────────────────────────────
    coloring = {}
    remaining_graph = Graph(set(graph.V),
                            [(u, v) for u, v in graph.E])
    color_id = 0

    while len(remaining_graph.V) > q and color_id < k:
        mis = get_mis(remaining_graph)
        if not mis:
            break
        # All members of the MIS get the same color (they are non-adjacent)
        for v in mis:
            coloring[v] = color_id
        color_id += 1

        # Remove MIS vertices from the residual and iterate
        new_V = remaining_graph.V - mis
        new_E = [(u, v) for u, v in remaining_graph.E
                 if u in new_V and v in new_V]
        remaining_graph = Graph(new_V, new_E)

        if not remaining_graph.V:
            break

        if remaining_graph.is_bipartite():
            # BFS 2-coloring for the bipartite residual
            color_map = {}
            for start in list(remaining_graph.V):
                if start in color_map:
                    continue
                queue = [start]
                color_map[start] = 0
                while queue:
                    u = queue.pop()
                    for w in remaining_graph.adj[u]:
                        if w not in color_map:
                            color_map[w] = 1 - color_map[u]
                            queue.append(w)
            for v, c in color_map.items():
                coloring[v] = color_id + c
            remaining_graph = Graph(set(), [])
            break

    # ── Tabu Search on residual (if any vertices remain uncolored) ────────────
    if remaining_graph.V:
        residual_k = max(2, k - color_id)
        res_col, _ = tabu_search(remaining_graph, residual_k)
        for v, c in res_col.items():
            coloring[v] = color_id + c

    # ── Expansion / legality check ────────────────────────────────────────────
    # Remap all assigned colors to [0, k-1] using modulo so conflicts are real
    unique_colors = sorted(set(coloring.values()))
    if len(unique_colors) <= k:
        # Map existing color ids to 0..len-1 compactly
        cmap = {c: i for i, c in enumerate(unique_colors)}
        coloring = {v: cmap[c] for v, c in coloring.items()}
    else:
        # More colors than k — force into k buckets (will create conflicts)
        coloring = {v: c % k for v, c in coloring.items()}

    f = fitness(coloring, graph)
    if f > 0:
        # Final Tabu Search pass on the full graph with exactly k colors
        coloring, f = tabu_search(graph, k, max_iter=len(graph.E) * 2)

    return coloring, f


# ─────────────────────────────────────────────────────────────────────────────
# Utility: find approximate chromatic number
# ─────────────────────────────────────────────────────────────────────────────
def chromatic_number(graph, k_max=20):
    """Try increasing values of k until a legal coloring is found."""
    # A graph with any edge needs at least 2 colors
    k_min = 2 if graph.E else 1
    for k in range(k_min, k_max + 1):
        coloring, f = tamisg(graph, k)
        if f == 0:
            return k, coloring
    return k_max, None


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────
def petersen_graph():
    """The Petersen graph — chromatic number = 3."""
    outer = list(range(5))
    inner = list(range(5, 10))
    edges  = [(i, (i+1) % 5) for i in outer]          # outer cycle
    edges += [(i+5, (i+2) % 5 + 5) for i in range(5)] # inner pentagram
    edges += [(i, i+5) for i in outer]                # spokes
    return Graph(list(range(10)), edges)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import networkx as nx

    random.seed(42)

    # ── Test 1: Petersen graph ────────────────────────────────────────────────
    g = petersen_graph()
    k, coloring = chromatic_number(g)
    print(f"Petersen graph — chromatic number ≈ {k}  (expected 3)")
    print(f"  conflicts = {fitness(coloring, g, k)}")
    print(f"  coloring  = {coloring}")

    # ── Test 2: Random graph ──────────────────────────────────────────────────
    random.seed(0)
    n_v = 15
    vertices = list(range(n_v))
    edges = [(i, j) for i in range(n_v) for j in range(i+1, n_v)
             if random.random() < 0.3]
    g2 = Graph(vertices, edges)
    k2, col2 = chromatic_number(g2)
    print(f"\nRandom graph (15 vtx, p=0.3) — chromatic number ≈ {k2}")
    print(f"  conflicts = {fitness(col2, g2, k2)}")

    # ── Visualise ─────────────────────────────────────────────────────────────
    palette = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
        "#9b59b6", "#1abc9c", "#e67e22", "#34495e"
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("TAMISG — Graph Coloring via MIS + Tabu Search\n"
                 "(Guzmán-Ponce et al., MICAI 2020)", fontsize=12)

    for ax, (g_obj, coloring_obj, title) in zip(axes, [
        (g, coloring, f"Petersen graph  (χ ≈ {k})"),
        (g2, col2,   f"Random graph    (χ ≈ {k2})")
    ]):
        G_nx = nx.Graph()
        G_nx.add_nodes_from(g_obj.V)
        G_nx.add_edges_from(g_obj.E)
        node_colors = [palette[coloring_obj[v] % len(palette)]
                       for v in G_nx.nodes()]
        pos = nx.spring_layout(G_nx, seed=42)
        nx.draw(G_nx, pos, ax=ax, with_labels=True, node_color=node_colors,
                node_size=600, font_color="white", font_weight="bold",
                edge_color="#aaa")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig("tamisg_result.png", dpi=150)
    plt.show()
    print("\nPlot saved as tamisg_result.png")
