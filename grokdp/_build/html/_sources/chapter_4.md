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

(chapter_4)=

# Intermediate Dynamic Programming Concepts

In this chapter, we'll explore more advanced Dynamic Programming problems, focusing on 1D and 2D DP concepts. These problems will help you understand how to approach more complex scenarios using DP techniques.

## 1D DP Problems

### Maximum Subarray Sum (Kadane's Algorithm)

#### Problem Statement:
Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

#### Solution:
This problem can be solved using Kadane's algorithm, which is a classic example of 1D dynamic programming.

```{code-cell} python3
    def max_subarray(nums):
        max_sum = current_sum = nums[0]
        for num in nums[1:]:
            current_sum = max(num, current_sum + num)
            max_sum = max(max_sum, current_sum)
        return max_sum

    # Example usage
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(max_subarray(nums))  # Output: 6
```

#### Explanation:
The key idea is to maintain two variables:
1. `current_sum`: the maximum sum ending at the current position
2. `max_sum`: the maximum sum seen so far

The recurrence relation can be expressed as:

$current\_sum[i] = \max(nums[i], current\_sum[i-1] + nums[i])$

$max\_sum = \max(max\_sum, current\_sum[i])$

This algorithm has a time complexity of O(n) and space complexity of O(1).

### Longest Increasing Subsequence

#### Problem Statement:
Given an integer array `nums`, return the length of the longest strictly increasing subsequence.

#### Solution:
We can solve this using dynamic programming with a time complexity of O(n^2).

```{code-cell} python3
    def longest_increasing_subsequence(nums):
        if not nums:
            return 0
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    # Example usage
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    print(longest_increasing_subsequence(nums))  # Output: 4
```

#### Explanation:
- `dp[i]` represents the length of the longest increasing subsequence ending at index i.
- The recurrence relation is:

  $dp[i] = \max(dp[i], dp[j] + 1)$ for all $j < i$ where $nums[i] > nums[j]$

- The final answer is the maximum value in the dp array.

## 2D DP Problems

### Grid Traveler Problem

#### Problem Statement:
Given a grid of size m x n, a traveler starts from the top-left corner and can only move right or down. The traveler wants to reach the bottom-right corner. How many possible unique paths are there?

#### Solution:
We can solve this using a 2D DP approach.

```{code-cell} python3
    def unique_paths(m, n):
        dp = [[1] * n for _ in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]

    # Example usage
    print(unique_paths(3, 7))  # Output: 28
```

#### Explanation:
- `dp[i][j]` represents the number of unique paths to reach the cell (i, j).
- The recurrence relation is:

  $dp[i][j] = dp[i-1][j] + dp[i][j-1]$

- We initialize the first row and first column to 1 since there's only one way to reach any cell in these regions.
- The final answer is in `dp[m-1][n-1]`.

### Longest Common Subsequence

#### Problem Statement:
Given two strings `text1` and `text2`, return the length of their longest common subsequence. If there is no common subsequence, return 0.

#### Solution:
This is a classic 2D DP problem.

```{code-cell} python3
    def longest_common_subsequence(text1, text2):
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]

    # Example usage
    text1 = "abcde"
    text2 = "ace"
    print(longest_common_subsequence(text1, text2))  # Output: 3
```

#### Explanation:
- `dp[i][j]` represents the length of the longest common subsequence of `text1[:i]` and `text2[:j]`.
- The recurrence relation is:
  
  If $text1[i-1] == text2[j-1]$:
    $dp[i][j] = dp[i-1][j-1] + 1$
  
  Else:
    $dp[i][j] = \max(dp[i-1][j], dp[i][j-1])$

- We initialize the first row and first column to 0.
- The final answer is in `dp[m][n]`.

These intermediate DP problems demonstrate how to apply DP concepts to more complex scenarios. They introduce the idea of using 1D and 2D arrays to store intermediate results, and show how to derive and apply more intricate recurrence relations.

In the next chapter, we'll explore even more advanced DP techniques and tackle some challenging problems that often appear in coding interviews.

