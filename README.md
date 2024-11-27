We consider the Minimum Weighted Crossings with Constraints Problem (MWCCP), which is a generalization of the Minimum Crossings Problem. In the MWCCP we are given an undirected weighted bipartite graph \( G = (U \cup V, E) \) with node sets \( U = \{1, \dots, m\} \) and \( V = \{m + 1, \dots, n\} \) corresponding to the two partitions, edge set \( E \), and a set of constraints \( C \). The node sets \( U \) and \( V \) are disjoint. The edges \( e = (u, v) \in E \subseteq U \times V \) have associated weights \( w_e = w_{u,v} \). Precedence constraints \( C \) are given in the form of a set of ordered node pairs \( C \subseteq V \times V \), where \( (v, v') \in C \) means that node \( v \) must appear before node \( v' \) in a feasible solution.

The nodes of the graph \( G \) are to be arranged in two layers. The first layer contains all nodes of set \( U \) in fixed label order \( 1, \dots, m \), while the second layer contains the nodes of set \( V \) in an order to be determined. The goal of the MWCCP is to find an ordering of the nodes in \( V \) such that the weighted edge crossings are minimized while satisfying all constraints \( C \).

A candidate solution is thus represented by a permutation \( \pi = (\pi_{m+1}, \dots, \pi_n) \) of the nodes in \( V \). It is only feasible if all of the constraints \( C \) are fulfilled, i.e., \( \text{pos}_\pi(v) < \text{pos}_\pi(v') \), \( \forall (v, v') \in C \), where \( \text{pos}(v) \) refers to the position of a node \( v \in V \) in the permutation \( \pi \).

The objective function to be minimized is:
\[
f(\pi) = \sum_{(u, v) \in E} \sum_{\substack{(u', v') \in E \\ u < u'}} 
(w_{u,v} + w_{u', v'}) \cdot \delta_{\pi}((u, v), (u', v'))
\]
where
\[
\delta_{\pi}((u, v), (u', v')) = 
\begin{cases} 
1 & \text{if } \text{pos}_{\pi}(v) > \text{pos}_{\pi}(v'), \\ 
0 & \text{otherwise.} 
\end{cases}
\]
