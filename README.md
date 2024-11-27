# Minimum Weighted Crossings with Constraints Problem (MWCCP)

The **Minimum Weighted Crossings with Constraints Problem (MWCCP)** is a generalization of the Minimum Crossings Problem. It is defined as follows:

## Problem Definition

We are given an **undirected weighted bipartite graph** $G = (U \cup V, E)$ with:
- **Node sets**:
  - $U = \{ 1, \dots, m \}$ (first partition)
  - $V = \{m + 1, \dots, n\}$ (second partition)
- $U $ and $V$ are disjoint.
- **Edge set**: \( E \subseteq U \times V \) (edges between the two partitions).
- **Edge weights**: \( w_e = w_{u,v} \) for \( e = (u, v) \in E \).

### Constraints
Precedence constraints \( C \) are given as a set of ordered node pairs:
\[
C \subseteq V \times V
\]
where \( (v, v') \in C \) means that **node \( v \) must appear before node \( v' \)** in a feasible solution.

### Goal
The nodes of the graph \( G \) are to be arranged in two layers:
1. The **first layer** contains all nodes in \( U \), arranged in fixed label order \( 1, \dots, m \).
2. The **second layer** contains all nodes in \( V \), arranged in an order to be determined.

The goal is to find an ordering of the nodes in \( V \) such that:
- The **weighted edge crossings** are minimized.
- All precedence constraints \( C \) are satisfied.

---

## Solution Representation

A candidate solution is represented by a **permutation** \( \pi = (\pi_{m+1}, \dots, \pi_n) \) of the nodes in \( V \).

### Feasibility
A solution is feasible if all precedence constraints \( C \) are fulfilled:
\[
\text{pos}_\pi(v) < \text{pos}_\pi(v'), \quad \forall (v, v') \in C
\]
where \( \text{pos}_\pi(v) \) is the position of node \( v \) in the permutation \( \pi \).

---

## Objective Function

The objective is to minimize the following function:
\[
f(\pi) = \sum_{(u, v) \in E} \sum_{\substack{(u', v') \in E \\ u < u'}} 
(w_{u,v} + w_{u', v'}) \cdot \delta_{\pi}((u, v), (u', v'))
\]

### Crossing Indicator
The indicator function \( \delta_{\pi}((u, v), (u', v')) \) is defined as:
\[
\delta_{\pi}((u, v), (u', v')) = 
\begin{cases} 
1 & \text{if } \text{pos}_{\pi}(v) > \text{pos}_{\pi}(v'), \\ 
0 & \text{otherwise.} 
\end{cases}
\]

---

This formulation ensures that the **weighted crossings** are minimized while satisfying all precedence constraints.
