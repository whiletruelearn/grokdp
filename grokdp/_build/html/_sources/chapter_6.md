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

(chapter_6)=

# Classic Dynamic Programming Problems

In this chapter, we'll explore some classic Dynamic Programming problems that are frequently asked in coding interviews and competitive programming contests. These problems are chosen for their educational value and the important DP concepts they illustrate.

## 1. Knapsack Problem

### Problem Statement:
Given a set of items, each with a weight and a value, determine the number of each item to include in a collection so that the total weight is less than or equal to a given limit and the total value is as large as possible.

### Solution:

```{code-cell} python3
    def knapsack(values, weights, capacity):
        n = len(values)
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
                else:
                    dp[i][w] = dp[i-1][w]
        
        return dp[n][capacity]

    # Example usage
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50
    print(knapsack(values, weights, capacity))  # Output: 220
```

### Explanation:
- We use a 2D DP table where `dp[i][w]` represents the maximum value that can be obtained using the first i items and with a maximum weight of w.
- The recurrence relation is:
  
  $dp[i][w] = \max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])$ if $weights[i-1] \leq w$
  
  $dp[i][w] = dp[i-1][w]$ otherwise

- Time Complexity: $O(n \times capacity)$
- Space Complexity: $O(n \times capacity)$

## 2. Edit Distance

### Problem Statement:
Given two strings `word1` and `word2`, return the minimum number of operations required to convert `word1` to `word2`. You have the following three operations permitted on a word:
- Insert a character
- Delete a character
- Replace a character

### Solution:

```{code-cell} python3
    def min_distance(word1, word2):
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j],    # Delete
                                       dp[i][j-1],    # Insert
                                       dp[i-1][j-1])  # Replace
        
        return dp[m][n]

    # Example usage
    word1 = "horse"
    word2 = "ros"
    print(min_distance(word1, word2))  # Output: 3
```

### Explanation:
- We use a 2D DP table where `dp[i][j]` represents the minimum number of operations to convert the first i characters of `word1` to the first j characters of `word2`.
- The recurrence relation is:
  
  If $word1[i-1] == word2[j-1]$:
    $dp[i][j] = dp[i-1][j-1]$
  
  Else:
    $dp[i][j] = 1 + \min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])$

- Time Complexity: $O(m \times n)$
- Space Complexity: $O(m \times n)$

## 3. Palindrome Partitioning

### Problem Statement:
Given a string s, partition s such that every substring of the partition is a palindrome. Return the minimum cuts needed for a palindrome partitioning of s.

### Solution:

```{code-cell} python3
    def min_cut(s):
        n = len(s)
        is_palindrome = [[False] * n for _ in range(n)]
        cut = [0] * n
        
        for i in range(n):
            min_cut = i
            for j in range(i + 1):
                if s[i] == s[j] and (i - j <= 2 or is_palindrome[j+1][i-1]):
                    is_palindrome[j][i] = True
                    min_cut = 0 if j == 0 else min(min_cut, cut[j-1] + 1)
            cut[i] = min_cut
        
        return cut[n-1]

    # Example usage
    s = "aab"
    print(min_cut(s))  # Output: 1
```

### Explanation:
- We use two DP tables:
  1. `is_palindrome[i][j]` to store whether the substring s[i:j+1] is a palindrome
  2. `cut[i]` to store the minimum number of cuts needed for the first i+1 characters
- We iterate through all possible ending positions and find the minimum number of cuts needed.
- The recurrence relation for `cut` is:
  
  $cut[i] = \min_{0 \leq j \leq i} \{cut[j-1] + 1\}$ if $s[j:i+1]$ is a palindrome

- Time Complexity: $O(n^2)$
- Space Complexity: $O(n^2)$

These classic DP problems demonstrate various techniques:
1. The Knapsack problem shows how to handle problems with weight constraints.
2. Edit Distance illustrates how to solve string manipulation problems using DP.
3. Palindrome Partitioning combines string manipulation with optimization.

Understanding these problems and their solutions will significantly improve your ability to recognize and solve DP problems in interviews and competitions. In the next chapter, we'll focus on DP problems specifically related to strings, which form a significant category of their own.