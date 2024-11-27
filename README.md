# Minimum Weighted Crossings with Constraints Problem (MWCCP)

The **Minimum Weighted Crossings with Constraints Problem (MWCCP)** is a generalization of the Minimum Crossings Problem.

## Problem Definition

We are given an **undirected weighted bipartite graph** $G = (U ∪ V, E)$ with:

- **Node sets**:
  - `U = {1, ..., m}`: the first partition.
  - `V = {m + 1, ..., n}`: the second partition.
- `U` and `V` are disjoint.
- **Edge set**: $E ⊆ U × V$, consisting of edges between the two partitions.
- **Edge weights**: $w_e = w_{u,v}$ for $e = (u, v) ∈ E$.

### Constraints

Precedence constraints $C$ are given as a set of ordered node pairs:

$C ⊆ V × V$,

where $(v, v') ∈ C$ means that **node $v$ must appear before node $v'$** in a feasible solution.

### Goal

The nodes of the graph $G$ are to be arranged in two layers:

1. The **first layer** contains all nodes in $U$, arranged in fixed label order `1, ..., m`.
2. The **second layer** contains all nodes in $V$, arranged in an order to be determined.

The goal is to find an ordering of the nodes in $V$ such that:

- The **weighted edge crossings** are minimized.
- All precedence constraints $C$ are satisfied.

---

## Solution Representation

A candidate solution is represented by a **permutation** $π = (π_{m+1}, ..., π_n)$ of the nodes in $V$.

### Feasibility

A solution is feasible if all precedence constraints $C$ are fulfilled:

$$
pos_π(v) < pos_π(v'), \forall (v, v') ∈ C
$$

where $pos_π(v)$ is the position of node $v$ in the permutation $π$.

---

## Objective Function

The objective is to minimize the following function:

$$
f(π) = ∑_{(u, v) ∈ E} ∑_{(u', v') ∈ E, u < u'} 
       (w_{u,v} + w_{u',v'}) · δ_π((u, v), (u', v'))
$$

### Crossing Indicator

The indicator function $δ_π((u, v), (u', v'))$ is defined as:

$δ_π((u, v), (u', v')) =$
- $1$, if $pos_π(v) > pos_π(v')$
- $0$, otherwise


This formulation ensures that the **weighted crossings** are minimized while satisfying all precedence constraints.

---
## Results

![Algorithms Comparison](https://github.com/bergio13/heuristic-optimization-MWCCP/blob/main/images/algo_comp.png)

