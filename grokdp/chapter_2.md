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

(chapter_2)=

# Fundamentals of Dynamic Programming

In this chapter, we'll dive deeper into the fundamental approaches used in Dynamic Programming (DP): memoization and tabulation. We'll also discuss how to analyze the time and space complexity of DP solutions.

## Memoization (Top-down approach)

Memoization is a top-down approach to dynamic programming where we start with the original problem and break it down into smaller subproblems. As we solve each subproblem, we store its result. If we encounter the same subproblem again, we can simply look up the previously computed result instead of recalculating it.

Let's implement the Fibonacci sequence using memoization:

```{code-cell} python3
    def fibonacci_memo(n, memo={}):
        if n in memo:
            return memo[n]
        if n <= 1:
            return n
        memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
        return memo[n]

    # Example usage
    print(fibonacci_memo(100))
```

In this implementation:
1. We use a dictionary `memo` to store computed results.
2. Before calculating `fibonacci_memo(n)`, we check if it's already in `memo`.
3. If it's not in `memo`, we calculate it and store the result.

This approach significantly reduces the number of recursive calls, improving efficiency.

## Tabulation (Bottom-up approach)

Tabulation is a bottom-up approach where we start by solving the smallest subproblems and use their solutions to build up to the solution of the original problem. This is typically implemented using iteration rather than recursion.

Here's the Fibonacci sequence implemented using tabulation:

```{code-cell} python3
    def fibonacci_tab(n):
        if n <= 1:
            return n
        dp = [0] * (n + 1)
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]

    # Example usage
    print(fibonacci_tab(100))
```

In this implementation:
1. We create an array `dp` to store all fibonacci numbers up to n.
2. We start with the base cases (dp[0] = 0, dp[1] = 1).
3. We iteratively fill the array, with each number being the sum of the two preceding ones.

## Time and Space Complexity Analysis

Understanding the time and space complexity of DP solutions is crucial for writing efficient algorithms.

### Time Complexity

For both memoization and tabulation approaches to the Fibonacci sequence:
- Time Complexity: O(n)
  - We perform a constant amount of work for each number from 0 to n.

### Space Complexity

- Memoization:
  - Space Complexity: O(n)
    - In the worst case, we store results for all numbers from 0 to n in the memo dictionary.
  - Call Stack Space: O(n)
    - The recursive calls can go n levels deep.

- Tabulation:
  - Space Complexity: O(n)
    - We explicitly create an array of size n+1.
  - Call Stack Space: O(1)
    - We use iteration, so there's no recursive call stack.

While both approaches have the same time and space complexity for the Fibonacci problem, they can differ for other problems. 

Memoization is often easier to implement as it follows the natural recursive structure of the problem. However, it can lead to stack overflow for very large inputs due to the depth of recursive calls.

Tabulation usually provides better space complexity as it doesn't use recursion, avoiding call stack issues. It can also be more efficient as the order of computation is more straightforward.

In the next chapter, we'll apply these techniques to solve some basic Dynamic Programming problems, which will help solidify your understanding of these concepts.