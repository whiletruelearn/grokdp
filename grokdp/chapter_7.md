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

(chapter_7)=

# Dynamic Programming in Strings

Dynamic Programming (DP) is a powerful technique that can be applied to various string problems. In this chapter, we'll explore two classic problems: the Longest Palindromic Subsequence and Regular Expression Matching.

## 7.1 Longest Palindromic Subsequence

### Problem Statement

Given a string, find the length of its longest palindromic subsequence. A palindromic subsequence is a subsequence that reads the same backwards as forwards.

For example, given the string "BBABCBCAB", the longest palindromic subsequence is "BABCBAB", which has a length of 7.

### Approach

We can solve this problem using a 2D DP table. Let's define $dp[i][j]$ as the length of the longest palindromic subsequence in the substring $s[i:j+1]$.

The recurrence relation is:

1. If $s[i] == s[j]$ and $i != j$: $dp[i][j] = dp[i+1][j-1] + 2$
2. If $s[i] == s[j]$ and $i == j$: $dp[i][j] = 1$
3. If $s[i] != s[j]$: $dp[i][j] = max(dp[i+1][j], dp[i][j-1])$

### Implementation

```{code-cell} python3
def longest_palindromic_subsequence(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    
    # Base case: palindromes of length 1
    for i in range(n):
        dp[i][i] = 1
    
    # Fill the dp table
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and length == 2:
                dp[i][j] = 2
            elif s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    
    return dp[0][n-1]

# Test the function
s = "BBABCBCAB"
print(f"Length of longest palindromic subsequence: {longest_palindromic_subsequence(s)}")
```

### Complexity Analysis

- Time Complexity: $O(n^2)$, where $n$ is the length of the string.
- Space Complexity: $O(n^2)$ to store the DP table.

### Visualization

Here's a text-based visualization of how the DP table would be filled for the string "BBAB":

```
    B   B   A   B
B   1   2   2   3
B       1   1   3
A           1   1
B               1
```

## 7.2 Regular Expression Matching

### Problem Statement

Implement regular expression matching with support for '.' and '*' where:
- '.' Matches any single character.
- '*' Matches zero or more of the preceding element.

The matching should cover the entire input string (not partial).

### Approach

We can solve this using a 2D DP table. Let $dp[i][j]$ be true if the first $i$ characters in the string match the first $j$ characters of the pattern.

The recurrence relation is:

1. If $p[j-1] == s[i-1]$ or $p[j-1] == '.'$: $dp[i][j] = dp[i-1][j-1]$
2. If $p[j-1] == '*'$:
   - $dp[i][j] = dp[i][j-2]$ (zero occurrence)
   - If $p[j-2] == s[i-1]$ or $p[j-2] == '.'$: $dp[i][j] |= dp[i-1][j]$ (one or more occurrences)

### Implementation

```{code-cell} python3
def is_match(s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    # Empty pattern matches empty string
    dp[0][0] = True
    
    # Patterns with '*' can match empty string
    for j in range(1, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
    
    # Fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == s[i-1] or p[j-1] == '.':
                dp[i][j] = dp[i-1][j-1]
            elif p[j-1] == '*':
                dp[i][j] = dp[i][j-2]
                if p[j-2] == s[i-1] or p[j-2] == '.':
                    dp[i][j] |= dp[i-1][j]
    
    return dp[m][n]

# Test the function
s = "aa"
p = "a*"
print(f"Does '{p}' match '{s}'? {is_match(s, p)}")
```

### Complexity Analysis

- Time Complexity: $O(mn)$, where $m$ and $n$ are the lengths of the string and pattern respectively.
- Space Complexity: $O(mn)$ to store the DP table.

### Visualization

Here's a text-based visualization of how the DP table would be filled for the string "aa" and pattern "a*":

```
    ε   a   *
ε   T   F   T
a   F   T   T
a   F   F   T
```

In this table, 'T' represents True (match) and 'F' represents False (no match).

## Conclusion

Dynamic Programming in strings often involves 2D DP tables and can solve complex pattern matching and subsequence problems efficiently. The key is to define the right recurrence relation and build the DP table step by step.

In the next chapter, we'll explore Dynamic Programming in Arrays and Matrices, which will build upon these concepts and introduce new techniques.
