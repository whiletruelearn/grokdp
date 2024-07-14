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

(chapter_11)=

# Chapter 11: Real-world Applications of Dynamic Programming

Dynamic Programming (DP) is not just a theoretical concept—it has numerous practical applications across various industries. In this chapter, we'll explore how DP is used to solve real-world problems in different domains.

## 11.1 Bioinformatics: Sequence Alignment

One of the most important applications of DP in bioinformatics is sequence alignment, used to compare DNA, RNA, or protein sequences.

### Problem: Global Sequence Alignment

Given two sequences, find the optimal alignment that maximizes similarity.

### Solution: Needleman-Wunsch Algorithm

```{code-cell} python3
def needleman_wunsch(seq1, seq2, match_score=1, mismatch_score=-1, gap_penalty=-1):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i * gap_penalty
    for j in range(n + 1):
        dp[0][j] = j * gap_penalty
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score)
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty
            dp[i][j] = max(match, delete, insert)
    
    return dp[m][n]

# Example usage
seq1 = "AGGCTATCACCTGACCTCCAGGCCGATGCCC"
seq2 = "TAGCTATCACGACCGCGGTCGATTTGCCCGAC"
alignment_score = needleman_wunsch(seq1, seq2)
print(f"Optimal alignment score: {alignment_score}")
```

This algorithm is widely used in bioinformatics for comparing genetic sequences, helping researchers understand evolutionary relationships and identify similar regions in different organisms.

## 11.2 Finance: Option Pricing

In financial mathematics, DP is used for option pricing, particularly for American options which can be exercised before the expiration date.

### Problem: American Option Pricing

Determine the fair price of an American put option.

### Solution: Binomial Option Pricing Model

```{code-cell} python3
import math

def american_put_option(S, K, T, r, sigma, N):
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    prices = [S * (d ** j) * (u ** (N - j)) for j in range(N + 1)]
    
    # Initialize option values at maturity
    values = [max(K - S, 0) for S in prices]
    
    # Backward induction
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            S = S * (u ** (i - j)) * (d ** j)
            hold_value = (p * values[j] + (1 - p) * values[j + 1]) * math.exp(-r * dt)
            exercise_value = max(K - S, 0)
            values[j] = max(hold_value, exercise_value)
    
    return values[0]

# Example usage
S = 100  # Current stock price
K = 100  # Strike price
T = 1    # Time to maturity (in years)
r = 0.05 # Risk-free interest rate
sigma = 0.2 # Volatility
N = 100  # Number of time steps

option_price = american_put_option(S, K, T, r, sigma, N)
print(f"Price of the American put option: ${option_price:.2f}")
```

This model helps financial institutions and investors accurately price options, manage risk, and make informed investment decisions.

## 11.3 Natural Language Processing: Speech Recognition

DP plays a crucial role in speech recognition systems, particularly in the process of decoding speech into text.

### Problem: Finding the Most Likely Sequence of Words

Given a sequence of acoustic features, find the most likely sequence of words that produced those features.

### Solution: Viterbi Algorithm

```{code-cell} python3
import numpy as np

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
    
    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]
    
    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        
        for y in states:
            (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        
        path = newpath
    
    # Find the most likely sequence
    (prob, state) = max((V[len(obs) - 1][y], y) for y in states)
    return (prob, path[state])

# Example usage (simplified)
states = ('Rainy', 'Sunny')
observations = ('walk', 'shop', 'clean')
start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
transition_probability = {
    'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
    'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
}
emission_probability = {
    'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
    'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
}

prob, path = viterbi(observations, states, start_probability, transition_probability, emission_probability)
print(f"Most likely weather sequence: {' '.join(path)}")
print(f"Probability: {prob}")
```

While this example uses a simple weather model, the same principle is applied in speech recognition systems to decode acoustic signals into text, helping power virtual assistants, transcription services, and more.

## 11.4 Robotics: Path Planning

DP is used in robotics for path planning, helping robots navigate efficiently through complex environments.

### Problem: Finding the Shortest Path in a Grid

Given a grid with obstacles, find the shortest path from start to goal.

### Solution: A* Algorithm (a DP-based heuristic search algorithm)

```{code-cell} python3
import heapq

def manhattan_distance(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

def a_star(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    heap = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan_distance(start, goal)}
    
    while heap:
        current = heapq.heappop(heap)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0]][neighbor[1]] == 0:
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + manhattan_distance(neighbor, goal)
                    heapq.heappush(heap, (f_score[neighbor], neighbor))
    
    return None  # No path found

# Example usage
grid = [
    [0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0]
]
start = (0, 0)
goal = (4, 4)

path = a_star(grid, start, goal)
if path:
    print(f"Shortest path: {path}")
else:
    print("No path found")
```

This algorithm helps robots navigate efficiently in various applications, from warehouse automation to autonomous vehicles.

## Conclusion

These examples demonstrate how Dynamic Programming is applied across diverse fields to solve complex real-world problems. From analyzing genetic sequences in bioinformatics to pricing financial instruments, from decoding speech to planning robot paths, DP proves to be a versatile and powerful technique.

As we've seen, the core principles of DP—breaking down problems into smaller subproblems and storing intermediate results—remain consistent across these applications. However, each domain requires careful problem formulation and often combines DP with domain-specific knowledge and heuristics.

In the next chapter, we'll explore common patterns and problem-solving strategies in DP, which will help you recognize and solve DP problems across various domains.