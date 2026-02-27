# TAMISG: Graph Coloring via MIS + Tabu Search

> **Article:** Guzmán-Ponce, A., et al. (2020). *A Metaheuristic Algorithm to Face the Graph Coloring Problem*. **MICAI 2020**, Springer LNAI 12469, pp. 196–208. [DOI: 10.1007/978-3-030-61705-9_17](https://doi.org/10.1007/978-3-030-61705-9_17)

## 📖 Algorithm Overview

**TAMISG** (Tabu search + MIS for Graph coloring) is a metaheuristic that approximates the **chromatic number** χ(G) — the minimum number of colors needed to color a graph so that no two adjacent vertices share a color.

It combines three strategies:

| Component | Role |
|-----------|------|
| **GetMIS** (Algorithm 2) | Extracts Maximal Independent Sets from non-articulation vertices |
| **Modified Tabu Search** (Algorithm 1) | Minimises coloring conflicts with a tabu list |
| **E2COL logic** | Extraction-and-Expansion framework |

### Algorithm Flow

```
1. Extraction Phase
   - Repeat until |residual graph| ≤ q = |V| − (|V|·60/100)
       • Build a MIS from non-articulation vertices (GetMIS)
       • Assign all MIS vertices the same color
       • Remove MIS from the graph
   - Stop early if the residual is bipartite (needs only 2 colors)

2. Initial Coloring of Residual
   - Apply modified Tabu Search on the remaining graph

3. Expansion Phase
   - Re-integrate previously extracted independent sets
   - If conflicts remain → run another Tabu Search pass on the full graph
```

### GetMIS (Algorithm 2)

```
Input: G = (V, E)
Output: MIS K

K = []
Choose non-articulation vertex vₑ with highest degree
Add vₑ to K
noNeighbors ← sorted non-neighbors of vₑ (ascending degree)
while noNeighbors:
    vₑ ← next vertex
    if vₑ is non-articulation: add to K; remove its neighbors
return K
```

### Tabu Search (Algorithm 1)

The fitness function counts conflicts:

```
f(s) = Σᵢ |{ {u,v} ∈ E : u ∈ Cᵢ, v ∈ Cᵢ }|
```

Each iteration moves a conflict vertex to a new color class. The tabu tenure is:

```
tl = Random(0,9) + α · f(s₀)     (α = 0.6)
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install numpy matplotlib networkx

# Run the demo
python tamisg.py
```

## 📊 What the Demo Does

1. Colors the **Petersen graph** (expected χ = 3).
2. Colors a **random graph** (15 vertices, edge probability 0.3).
3. Prints the approximate chromatic number and conflict count for each.
4. Saves a side-by-side graph layout as **`tamisg_result.png`** with vertices color-coded.



## 📐 Key Functions

| Function | Description |
|----------|-------------|
| `get_mis(graph)` | Builds a Maximal Independent Set (Algorithm 2) |
| `tabu_search(graph, k)` | Modified Tabucol search (Algorithm 1) |
| `tamisg(graph, k)` | Main TAMISG coloring for a given k |
| `chromatic_number(graph)` | Tries increasing k until a legal coloring is found |

## 📋 Dependencies

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical utilities |
| `matplotlib` | Visualisation |
| `networkx` | Graph layout for plotting |
