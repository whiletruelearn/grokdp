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

# Basic Dynamic Programming Problems

In this chapter, we'll solve some basic Dynamic Programming problems. These problems will help you understand how to apply the concepts of memoization and tabulation that we learned in the previous chapter.

## 1. Fibonacci Sequence

We've already seen the Fibonacci sequence in previous chapters, but let's quickly recap it here as it's a classic example of a DP problem.

### Problem Statement:
Given a number n, find the nth Fibonacci number. The Fibonacci sequence is defined as:

$F(n) = F(n-1) + F(n-2)$, where $F(0) = 0$ and $F(1) = 1$.

### Solution:
We've already seen both memoization and tabulation approaches for this problem in Chapter 2. Here's the tabulation approach again for reference:

```{code-cell} python3
    def fibonacci(n):
        if n <= 1:
            return n
        dp = [0] * (n + 1)
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]

    # Example usage
    print(fibonacci(10))  # Output: 55
```

## 2. Climbing Stairs

### Problem Statement:
You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

### Solution:
This problem is similar to the Fibonacci sequence. At any step, you can arrive either from one step below or two steps below. We can represent this as:

$dp[i] = dp[i-1] + dp[i-2]$

Where $dp[i]$ is the number of ways to reach the i-th step.

```{code-cell} python3
    def climb_stairs(n):
        if n <= 2:
            return n
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]

    # Example usage
    print(climb_stairs(5))  # Output: 8
```

### Explanation:
- $dp[i]$ represents the number of ways to reach the i-th step.
- To reach the i-th step, we can either take a single step from the (i-1)th step or take two steps from the (i-2)th step.
- Therefore, $dp[i] = dp[i-1] + dp[i-2]$

## 3. Coin Change Problem

### Problem Statement:
Given an array of coin denominations and a target amount, find the minimum number of coins needed to make up that amount. If the amount cannot be made up by any combination of the coins, return -1.

### Solution:
We'll use a bottom-up (tabulation) approach for this problem. The recurrence relation can be expressed as:

$dp[i] = \min_{c \in coins} \{dp[i-c] + 1\}$ if $i \geq c$

Where $dp[i]$ is the minimum number of coins needed to make amount $i$.

```{code-cell} python3
    def coin_change(coins, amount):
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1

    # Example usage
    coins = [1, 2, 5]
    amount = 11
    print(coin_change(coins, amount))  # Output: 3 (5 + 5 + 1)
```


### Explanation:
- $dp[i]$ represents the minimum number of coins needed to make amount $i$.
- We initialize $dp[0] = 0$ (it takes 0 coins to make amount 0) and the rest to infinity.
- For each amount $i$ from 1 to the target amount:
  - For each coin denomination $c$:
    - If the coin value is less than or equal to the current amount, we have two choices:
      1. Don't use this coin (keep $dp[i]$ as is)
      2. Use this coin ($1 + dp[i - c]$)
    - We take the minimum of these two choices.
- At the end, $dp[amount]$ gives us the minimum number of coins needed.

These problems demonstrate how Dynamic Programming can be applied to solve various types of questions. They all follow the same pattern:
1. Define the subproblems
2. Find the recurrence relation between subproblems
3. Solve the base cases
4. Either use memoization (top-down) or build a table (bottom-up) to solve larger problems

As you practice more DP problems, you'll start recognizing these patterns more easily. In the next chapter, we'll dive into more complex DP problems and explore different types of DP patterns.