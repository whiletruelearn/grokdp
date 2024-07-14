---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(chapter_9)=

# Chapter 9: Dynamic Programming with Graphs

Dynamic Programming (DP) can be effectively applied to various graph problems. In this chapter, we'll explore two classic problems: the All-Pairs Shortest Path problem using the Floyd-Warshall algorithm and the Traveling Salesman Problem.

## 9.1 All-Pairs Shortest Path (Floyd-Warshall Algorithm)

### Problem Statement

Given a weighted graph, find the shortest path between every pair of vertices. The graph may contain negative edge weights, but no negative-weight cycles.

### Approach

The Floyd-Warshall algorithm uses a 3D DP approach to solve this problem. Let $dp[k][i][j]$ represent the shortest path from vertex $i$ to vertex $j$ using vertices only from the set $\{0, 1, ..., k\}$ as intermediate vertices.

The recurrence relation is:

$dp[k][i][j] = \min(dp[k-1][i][j], dp[k-1][i][k] + dp[k-1][k][j])$

This can be optimized to use only a 2D array by updating in-place.

### Implementation

```{code-cell} python3
def floyd_warshall(graph):
    n = len(graph)
    dp = [row[:] for row in graph]  # Create a copy of the graph
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j])
    
    return dp

# Test the function
INF = float('inf')
graph = [
    [0, 5, INF, 10],
    [INF, 0, 3, INF],
    [INF, INF, 0, 1],
    [INF, INF, INF, 0]
]

result = floyd_warshall(graph)
print("All-Pairs Shortest Paths:")
for row in result:
    print([x if x != INF else "INF" for x in row])
```

### Complexity Analysis

- Time Complexity: $O(n^3)$, where $n$ is the number of vertices.
- Space Complexity: $O(n^2)$ to store the DP table.

### Visualization

Here's a text-based visualization of how the DP table would be filled for the given graph:

```
Initial:        After k=0:      After k=1:      Final (k=3):
0    5    INF  10     0    5    INF  10     0    5    8    10     0    5    8    9
INF  0    3    INF    INF  0    3    INF    INF  0    3    INF    INF  0    3    4
INF  INF  0    1      INF  INF  0    1      INF  INF  0    1      INF  INF  0    1
INF  INF  INF  0      INF  INF  INF  0      INF  INF  INF  0      INF  INF  INF  0
```

## 9.2 Traveling Salesman Problem

### Problem Statement

Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?

### Approach

We can solve this problem using a DP approach with bitmasks. Let $dp[mask][i]$ represent the shortest path that visits all cities in the bitmask and ends at city $i$.

The recurrence relation is:

$dp[mask][i] = \min_{j \neq i, j \in mask} (dp[mask \setminus \{i\}][j] + dist[j][i])$

### Implementation

```python3
from itertools import combinations

def traveling_salesman(dist):
    n = len(dist)
    all_sets = []
    for r in range(1, n):
        all_sets.extend(combinations(range(1, n), r))
    
    # Initialize DP table
    dp = {}
    for i in range(1, n):
        dp[(1 << i, i)] = (dist[0][i], 0)
    
    # Iterate over all subsets of cities
    for subset in all_sets:
        mask = 0
        for bit in subset:
            mask |= 1 << bit
        
        for last in subset:
            prev = mask & ~(1 << last)
            dp[(mask, last)] = min(
                (dp[(prev, j)][0] + dist[j][last], j)
                for j in subset if j != last
            )
    
    # Find optimal tour
    mask = (1 << n) - 1
    optimal_tour = min(
        (dp[(mask, i)][0] + dist[i][0], i)
        for i in range(1, n)
    )
    
    return optimal_tour[0]

# Test the function
dist = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

print(f"Shortest tour length: {traveling_salesman(dist)}")
```

### Complexity Analysis

- Time Complexity: $O(n^2 2^n)$, where $n$ is the number of cities.
- Space Complexity: $O(n 2^n)$ to store the DP table.

### Visualization

For the Traveling Salesman Problem, visualizing the DP table is challenging due to its high dimensionality. Instead, let's visualize a simple example of how the optimal tour is constructed:

```
Cities: A, B, C, D

Step 1: A → B (10)
Step 2: B → D (25)
Step 3: D → C (30)
Step 4: C → A (15)

Total Distance: 10 + 25 + 30 + 15 = 80
```

## Conclusion

Dynamic Programming in graphs often involves innovative ways to represent states and transitions. The Floyd-Warshall algorithm demonstrates how DP can efficiently solve the all-pairs shortest path problem in $O(n^3)$ time, which is remarkable considering the problem's complexity.

The Traveling Salesman Problem, while NP-hard, becomes solvable for small to medium-sized inputs using DP with bitmasks. This approach showcases how DP can be combined with other techniques (in this case, bit manipulation) to solve complex optimization problems.

These techniques form the foundation for solving many real-world problems in logistics, network design, and operations research. In the next chapter, we'll explore optimization techniques in Dynamic Programming to make our solutions even more efficient.