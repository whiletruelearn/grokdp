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

(chapter_10)=

# Chapter 10: Optimization Techniques in Dynamic Programming

While Dynamic Programming (DP) is a powerful technique for solving complex problems, it can sometimes lead to solutions that are inefficient in terms of time or space complexity. In this chapter, we'll explore several optimization techniques that can make our DP solutions more efficient.

## 10.1 Space Optimization

One common issue with DP solutions is that they often use a lot of memory. However, in many cases, we can optimize the space usage without affecting the time complexity.

### Example: Fibonacci Sequence

Let's start with a simple example: calculating the nth Fibonacci number.

#### Naive DP Solution:

```{code-cell} python3
def fibonacci(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

print(fibonacci(10))  # Output: 55
```

This solution uses O(n) space.

#### Space-Optimized Solution:

```{code-cell} python3
def fibonacci_optimized(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

print(fibonacci_optimized(10))  # Output: 55
```

This optimized solution uses only O(1) space.

### Technique: Rolling Array

For problems where the current state depends only on a fixed number of previous states, we can use a "rolling array" to save space.

Example: Consider the climbing stairs problem where you can take 1, 2, or 3 steps at a time.

```{code-cell} python3
def climb_stairs(n):
    if n <= 2:
        return n
    dp = [0, 1, 2, 4]  # Base cases for n = 0, 1, 2, 3
    for i in range(4, n + 1):
        dp[i % 4] = dp[(i-1) % 4] + dp[(i-2) % 4] + dp[(i-3) % 4]
    return dp[n % 4]

print(climb_stairs(5))  # Output: 13
```

This solution uses only O(1) space instead of O(n).

## 10.2 Using Less State

Sometimes, we can reduce the dimensions of our DP table by clever problem analysis.

### Example: Knapsack Problem

Consider the 0/1 Knapsack problem where we need to maximize the value of items we can carry in a knapsack of capacity W.

#### Standard 2D DP Solution:

```{code-cell} python3
def knapsack(values, weights, W):
    n = len(values)
    dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, W + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][W]

values = [60, 100, 120]
weights = [10, 20, 30]
W = 50
print(knapsack(values, weights, W))  # Output: 220
```

This solution uses O(nW) space.

#### 1D DP Solution:

```{code-cell} python3
def knapsack_1d(values, weights, W):
    n = len(values)
    dp = [0] * (W + 1)
    
    for i in range(n):
        for w in range(W, weights[i] - 1, -1):
            dp[w] = max(dp[w], values[i] + dp[w - weights[i]])
    
    return dp[W]

values = [60, 100, 120]
weights = [10, 20, 30]
W = 50
print(knapsack_1d(values, weights, W))  # Output: 220
```

This optimized solution uses only O(W) space.

## 10.3 Combining Top-down and Bottom-up Approaches

Sometimes, a hybrid approach combining top-down (memoization) and bottom-up (tabulation) can be more efficient.

### Example: Matrix Chain Multiplication

Let's optimize the Matrix Chain Multiplication problem from the previous chapter.

```{code-cell} python3
def matrix_chain_multiplication(dimensions):
    n = len(dimensions) - 1
    memo = {}
    
    def dp(i, j):
        if i == j:
            return 0
        if (i, j) in memo:
            return memo[(i, j)]
        
        memo[(i, j)] = min(dp(i, k) + dp(k+1, j) + dimensions[i-1]*dimensions[k]*dimensions[j]
                           for k in range(i, j))
        return memo[(i, j)]
    
    return dp(1, n)

dimensions = [10, 30, 5, 60]
print(f"Minimum number of multiplications: {matrix_chain_multiplication(dimensions)}")
# Output: Minimum number of multiplications: 4500
```

This solution combines the top-down approach (recursive calls) with memoization, which can be more intuitive and sometimes more efficient than the purely bottom-up approach.

## Conclusion

Optimizing Dynamic Programming solutions often involves trade-offs between time and space complexity. The techniques we've covered - space optimization, using less state, and combining top-down and bottom-up approaches - can significantly improve the efficiency of our algorithms.

Remember, the best optimization technique depends on the specific problem and constraints. Always analyze your problem carefully to determine which optimization methods are most appropriate.

In the next chapter, we'll explore real-world applications of Dynamic Programming, seeing how these techniques are used in various domains.