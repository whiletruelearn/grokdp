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

(appendix)=

# Appendix

### Appendix A. Python Tips and Tricks for DP

Dynamic Programming (DP) can sometimes be challenging to implement. Here are some Python-specific tips and tricks to help you write cleaner, more efficient DP code.

#### 1. Use `functools.lru_cache` for Memoization

Python's `functools.lru_cache` is a decorator that can be used to automatically cache the results of function calls, making memoization straightforward.

```{code-cell} python3
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

print(fib(10))  # Output: 55
```

#### 2. Use Default Dictionaries for Dynamic Programming Tables

`collections.defaultdict` can simplify the initialization of DP tables, especially when dealing with multi-dimensional DP problems.

```{code-cell} python3
from collections import defaultdict

dp = defaultdict(int)
dp[0] = 1
dp[1] = 1

for i in range(2, 10):
    dp[i] = dp[i-1] + dp[i-2]

print(dp[9])  # Output: 55
```

#### 3. Inline Conditionals and List Comprehensions

Python's inline conditionals and list comprehensions can make your DP code more concise and readable.

```{code-cell} python3
def climb_stairs(n):
    if n <= 1:
        return 1
    dp = [0] * (n + 1)
    dp[0], dp[1] = 1, 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

print(climb_stairs(5))  # Output: 8
```

#### 4. Use Tuple Keys for Multi-Dimensional Problems

When working with multi-dimensional DP problems, you can use tuples as dictionary keys to represent states.

```{code-cell} python3
def grid_traveler(m, n):
    memo = {}
    def travel(m, n):
        if (m, n) in memo:
            return memo[(m, n)]
        if m == 0 or n == 0:
            return 0
        if m == 1 and n == 1:
            return 1
        memo[(m, n)] = travel(m-1, n) + travel(m, n-1)
        return memo[(m, n)]
    
    return travel(m, n)

print(grid_traveler(3, 3))  # Output: 6
```

---

### Appendix B. Common DP Patterns Cheat Sheet

Understanding common DP patterns can help you identify and solve DP problems more effectively. Here are some frequently encountered patterns:

#### 1. Fibonacci Sequence

Pattern: Simple recurrence relation with two previous states.

```{code-cell} python3
def fib(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

print(fib(10))  # Output: 55
```

#### 2. Climbing Stairs

Pattern: Similar to Fibonacci, but can be generalized to more steps.

```{code-cell} python3
def climb_stairs(n):
    if n <= 1:
        return 1
    dp = [0] * (n + 1)
    dp[0], dp[1] = 1, 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

print(climb_stairs(5))  # Output: 8
```

#### 3. Coin Change Problem

Pattern: Combinatorial problems with multiple options per state.

```{code-cell} python3
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

print(coin_change([1, 2, 5], 11))  # Output: 3
```

#### 4. Longest Increasing Subsequence

Pattern: Subsequence problems where order matters.

```{code-cell} python3
def length_of_lis(nums):
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

print(length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]))  # Output: 4
```

#### 5. Longest Common Subsequence

Pattern: Subsequence problems in two sequences.

```{code-cell} python3
def longest_common_subsequence(text1, text2):
    dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
    for i in range(1, len(text1) + 1):
        for j in range(1, len(text2) + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]

print(longest_common_subsequence("abcde", "ace"))  # Output: 3
```

By understanding these common patterns and their implementations, you can more effectively tackle a wide range of DP problems.