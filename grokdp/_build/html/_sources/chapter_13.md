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

(chapter_13)=

# Chapter 13: Practice Problems and Solutions

In this chapter, we'll work through a collection of Dynamic Programming problems that have been asked in actual coding interviews. For each problem, we'll provide a problem statement, an approach, a Python implementation, and a detailed explanation.

## Problem 1: House Robber

### Problem Statement

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array `nums` representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

### Approach

This is a classic DP problem. We can define our DP state as:

`dp[i]` = the maximum amount of money we can rob from the first i houses

The recurrence relation is:

`dp[i] = max(dp[i-1], dp[i-2] + nums[i])`

This means that for each house, we have two options:
1. Don't rob this house: `dp[i-1]`
2. Rob this house: `dp[i-2] + nums[i]` (we add the current house value to the max we could rob from two houses ago)

We take the maximum of these two options.

### Implementation

```{code-cell} python3
def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    dp = [0] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    
    return dp[-1]

# Test the function
print(rob([1,2,3,1]))  # Output: 4
print(rob([2,7,9,3,1]))  # Output: 12
```

### Explanation

- We start by handling edge cases: if there are no houses or only one house.
- We initialize our DP array. `dp[0]` is just the value of the first house, and `dp[1]` is the max of the first two houses.
- We then iterate through the rest of the houses, applying our recurrence relation.
- The final answer is the last element in our DP array, representing the maximum amount we can rob from all houses.

## Problem 2: Coin Change

### Problem Statement

You are given an integer array `coins` representing coins of different denominations and an integer `amount` representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

### Approach

We can define our DP state as:

`dp[i]` = the minimum number of coins needed to make amount i

The recurrence relation is:

`dp[i] = min(dp[i], dp[i - coin] + 1) for coin in coins if i >= coin`

This means that for each amount, we consider using each coin, and take the minimum number of coins needed.

### Implementation

```{code-cell} python3
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Test the function
print(coinChange([1,2,5], 11))  # Output: 3
print(coinChange([2], 3))  # Output: -1
```

### Explanation

- We initialize our DP array with infinity for all amounts, except 0 which requires 0 coins.
- We iterate through all amounts from 1 to our target amount.
- For each amount, we consider using each coin denomination.
- If the coin value is less than or equal to our current amount, we can use it. We update our DP value to be the minimum of its current value and the number of coins needed if we use the current coin.
- At the end, if `dp[amount]` is still infinity, it means we couldn't make up the amount with the given coins, so we return -1. Otherwise, we return `dp[amount]`.

## Problem 3: Longest Common Subsequence

### Problem Statement

Given two strings `text1` and `text2`, return the length of their longest common subsequence. If there is no common subsequence, return 0.

A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

A common subsequence of two strings is a subsequence that is common to both strings.

### Approach

We can define our DP state as:

`dp[i][j]` = the length of the longest common subsequence of text1[:i] and text2[:j]

The recurrence relation is:

If `text1[i-1] == text2[j-1]`: `dp[i][j] = dp[i-1][j-1] + 1`
Else: `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`

### Implementation

```{code-cell} python3
def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# Test the function
print(longestCommonSubsequence("abcde", "ace"))  # Output: 3
print(longestCommonSubsequence("abc", "def"))  # Output: 0
```

### Explanation

- We create a 2D DP table. `dp[i][j]` represents the length of the LCS of the first i characters of text1 and the first j characters of text2.
- We iterate through both strings character by character.
- If the characters match, we add 1 to the LCS length of the strings without these characters.
- If they don't match, we take the maximum of the LCS without the current character from either string.
- The final answer is in `dp[m][n]`, representing the LCS of the entire strings.



## Problem 4: Maximum Product Subarray

### Problem Statement
Given an integer array `nums`, find a contiguous non-empty subarray within the array that has the largest product, and return the product.

### Solution

This problem can be solved using a 1D Dynamic Programming approach. The key insight is to keep track of both the maximum and minimum product ending at each position, as negative numbers can affect the result.

```{code-cell} python3
def maxProduct(nums):
    if not nums:
        return 0
    
    max_so_far = min_so_far = result = nums[0]
    
    for i in range(1, len(nums)):
        temp_max = max(nums[i], max_so_far * nums[i], min_so_far * nums[i])
        min_so_far = min(nums[i], max_so_far * nums[i], min_so_far * nums[i])
        max_so_far = temp_max
        result = max(result, max_so_far)
    
    return result
```

### Explanation

1. We initialize `max_so_far`, `min_so_far`, and `result` with the first element of the array.
2. For each subsequent element, we calculate:
   - The maximum product ending at the current position
   - The minimum product ending at the current position
3. We update `result` if we find a larger product.

The time complexity is O(n), and the space complexity is O(1).

## Problem 5: Longest Increasing Subsequence

### Problem Statement
Given an integer array `nums`, return the length of the longest strictly increasing subsequence.

### Solution

We can solve this problem using a 1D DP approach.

```{code-cell} python3
def lengthOfLIS(nums):
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)
```

### Explanation

1. We initialize a DP array where `dp[i]` represents the length of the longest increasing subsequence ending at index `i`.
2. For each element, we look at all previous elements:
   - If the current element is greater than a previous element, we can potentially extend that subsequence.
   - We update `dp[i]` to be the maximum of its current value and the length of the subsequence ending at `j` plus 1.
3. The maximum value in the DP array is our answer.

The time complexity is O(n^2), and the space complexity is O(n).

## Problem 6: Edit Distance

### Problem Statement
Given two strings `word1` and `word2`, return the minimum number of operations required to convert `word1` to `word2`. You have the following three operations permitted on a word:
- Insert a character
- Delete a character
- Replace a character

### Solution

This problem can be solved using a 2D DP approach.

```{code-cell} python3
def minDistance(word1, word2):
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
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    return dp[m][n]
```

### Explanation

1. We create a 2D DP table where `dp[i][j]` represents the minimum number of operations to convert the first `i` characters of `word1` to the first `j` characters of `word2`.
2. We initialize the first row and column to represent the operations needed to convert to an empty string.
3. For each cell, we have two cases:
   - If the characters match, we take the value from the diagonal (no operation needed).
   - If they don't match, we take the minimum of insert, delete, or replace operations and add 1.
4. The bottom-right cell gives us the minimum number of operations needed.

The time and space complexity are both O(mn).

## Problem 7: Palindrome Partitioning II

### Problem Statement
Given a string `s`, partition `s` such that every substring of the partition is a palindrome. Return the minimum cuts needed for a palindrome partitioning of `s`.

### Solution

We can solve this problem using two DP tables.

```{code-cell} python3
def minCut(s):
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
```

### Explanation

1. We use a 2D DP table `is_palindrome` to store whether substrings are palindromes.
2. We use a 1D DP table `cut` to store the minimum number of cuts for each prefix of the string.
3. For each ending position `i`, we consider all possible starting positions `j`:
   - If the substring from `j` to `i` is a palindrome, we update `cut[i]`.
4. The final answer is in `cut[n-1]`.

The time and space complexity are both O(n^2).

## Problem 8: Burst Balloons

### Problem Statement
Given `n` balloons, indexed from 0 to `n-1`. Each balloon is painted with a number on it represented by array `nums`. You are asked to burst all the balloons. If you burst balloon `i` you will get `nums[i-1] * nums[i] * nums[i+1]` coins. If `i-1` or `i+1` goes out of bounds of the array, then treat it as if there is a balloon with a 1 painted on it.

Return the maximum coins you can collect by bursting the balloons wisely.

### Solution

This problem can be solved using a 2D DP approach.

```{code-cell} python3
def maxCoins(nums):
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    
    for length in range(2, n):
        for left in range(n - length):
            right = left + length
            for i in range(left + 1, right):
                coins = nums[left] * nums[i] * nums[right]
                dp[left][right] = max(dp[left][right], 
                                      dp[left][i] + coins + dp[i][right])
    
    return dp[0][n-1]
```

### Explanation

1. We add 1s to the start and end of the array to handle edge cases.
2. `dp[left][right]` represents the maximum coins obtained by bursting all balloons between `left` and `right`, exclusive.
3. We iterate over all possible lengths and left boundaries.
4. For each subproblem, we try all possible last balloons to burst and choose the maximum.

The time complexity is O(n^3), and the space complexity is O(n^2).

## Problem 9: Regular Expression Matching

### Problem Statement
Given an input string `s` and a pattern `p`, implement regular expression matching with support for '.' and '*' where:
- '.' Matches any single character.
- '*' Matches zero or more of the preceding element.

The matching should cover the entire input string (not partial).

### Solution

We can solve this problem using a 2D DP approach.

```{code-cell} python3
def isMatch(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    for j in range(1, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[i][j] = dp[i][j-2]
                if p[j-2] == '.' or p[j-2] == s[i-1]:
                    dp[i][j] |= dp[i-1][j]
            elif p[j-1] == '.' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]
    
    return dp[m][n]
```

### Explanation

1. We create a 2D DP table where `dp[i][j]` represents whether the first `i` characters of `s` match the first `j` characters of `p`.
2. We handle the base cases for empty strings.
3. For each cell, we consider different cases based on the current character in the pattern:
   - If it's '*', we can either ignore it or use it to match characters.
   - If it's '.' or matches the current character in s, we take the result from the previous state.

The time and space complexity are both O(mn).

## Problem 10: Interleaving String

### Problem Statement
Given strings `s1`, `s2`, and `s3`, find whether `s3` is formed by an interleaving of `s1` and `s2`.

An interleaving of two strings `s` and `t` is a configuration where `s` and `t` are divided into `n` and `m` substrings respectively, such that:

- `s = s1 + s2 + ... + sn`
- `t = t1 + t2 + ... + tm`
- `|n - m| <= 1`
- The interleaving is `s1 + t1 + s2 + t2 + s3 + t3 + ...` or `t1 + s1 + t2 + s2 + t3 + s3 + ...`

### Solution

We can solve this problem using a 2D DP approach.

```{code-cell} python3
def isInterleave(s1, s2, s3):
    m, n = len(s1), len(s2)
    if m + n != len(s3):
        return False
    
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i > 0:
                dp[i][j] |= dp[i-1][j] and s1[i-1] == s3[i+j-1]
            if j > 0:
                dp[i][j] |= dp[i][j-1] and s2[j-1] == s3[i+j-1]
    
    return dp[m][n]
```

### Explanation

1. We first check if the lengths match.
2. We create a 2D DP table where `dp[i][j]` represents whether the first `i` characters of `s1` and the first `j` characters of `s2` can interleave to form the first `i+j` characters of `s3`.
3. We initialize the base case for empty strings.
4. For each cell, we check if we can form the current interleaving by:
   - Using the current character from `s1` if it matches `s3`
   - Using the current character from `s2` if it matches `s3`
5. The final answer is in `dp[m][n]`.

The time and space complexity are both O(mn).



## Problem 11: Longest Palindromic Substring

### Problem Statement
Given a string `s`, return the longest palindromic substring in `s`.

### Solution

We can solve this problem using a 2D DP approach.

```{code-cell} python3
def longestPalindrome(s):
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    start, max_length = 0, 1
    
    # All substrings of length 1 are palindromes
    for i in range(n):
        dp[i][i] = True
    
    # Check for substrings of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_length = 2
    
    # Check for lengths greater than 2
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                if length > max_length:
                    start = i
                    max_length = length
    
    return s[start:start + max_length]
```

### Explanation

1. We use a 2D DP table where `dp[i][j]` represents whether the substring from index `i` to `j` is a palindrome.
2. We initialize palindromes of length 1 and 2.
3. For longer palindromes, we check if the first and last characters match and if the inner substring is a palindrome.
4. We keep track of the start index and length of the longest palindrome found.

Time complexity: O(n^2), Space complexity: O(n^2)

## Problem 12: Minimum Path Sum

### Problem Statement
Given a `m x n` grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path. You can only move either down or right at any point in time.

### Solution

We can solve this using a 2D DP approach.

```{code-cell} python3
def minPathSum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    
    dp[0][0] = grid[0][0]
    
    # Initialize first row
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    
    # Initialize first column
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    
    # Fill the dp table
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    
    return dp[m-1][n-1]
```

### Explanation

1. We create a 2D DP table where `dp[i][j]` represents the minimum path sum to reach cell (i, j).
2. We initialize the first row and column, as there's only one way to reach these cells.
3. For other cells, we take the minimum of the path from above and left, and add the current cell's value.
4. The bottom-right cell gives the minimum path sum.

Time complexity: O(mn), Space complexity: O(mn)

## Problem 13: Unique Binary Search Trees

### Problem Statement
Given an integer `n`, return the number of structurally unique BST's (binary search trees) which have exactly `n` nodes of unique values from 1 to n.

### Solution

We can solve this using a 1D DP approach.

```{code-cell} python3
def numTrees(n):
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    
    for i in range(2, n + 1):
        for j in range(1, i + 1):
            dp[i] += dp[j - 1] * dp[i - j]
    
    return dp[n]
```

### Explanation

1. We use a 1D DP array where `dp[i]` represents the number of unique BSTs with `i` nodes.
2. We initialize base cases for 0 and 1 node.
3. For `i` nodes, we consider each number as the root and multiply the number of possible left subtrees with right subtrees.
4. The sum of all these possibilities gives the total number of unique BSTs.

Time complexity: O(n^2), Space complexity: O(n)

## Problem 14: Best Time to Buy and Sell Stock with Cooldown

### Problem Statement
You are given an array `prices` where `prices[i]` is the price of a given stock on the ith day. Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times) with the following restrictions:

- After you sell your stock, you cannot buy stock on the next day (i.e., cooldown one day).
- You cannot sell and buy multiple times on a single day.

### Solution

We can solve this using a state machine DP approach.

```{code-cell} python3
def maxProfit(prices):
    if not prices:
        return 0
    
    n = len(prices)
    buy = [0] * n
    sell = [0] * n
    cool = [0] * n
    
    buy[0] = -prices[0]
    
    for i in range(1, n):
        buy[i] = max(buy[i-1], cool[i-1] - prices[i])
        sell[i] = max(sell[i-1], buy[i-1] + prices[i])
        cool[i] = max(cool[i-1], sell[i-1])
    
    return max(sell[-1], cool[-1])
```

### Explanation

1. We use three DP arrays to represent different states: `buy`, `sell`, and `cool`.
2. For each day, we update these states:
   - `buy[i]`: max of (not buying, buying after cooldown)
   - `sell[i]`: max of (not selling, selling what we bought)
   - `cool[i]`: max of (staying in cooldown, entering cooldown after selling)
3. The maximum of the last sell or cooldown state gives the maximum profit.

Time complexity: O(n), Space complexity: O(n)

## Problem 15: Partition Equal Subset Sum

### Problem Statement
Given a non-empty array `nums` containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.

### Solution

We can solve this using a 1D DP approach.

```{code-cell} python3
def canPartition(nums):
    total_sum = sum(nums)
    if total_sum % 2 != 0:
        return False
    
    target = total_sum // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] |= dp[i - num]
    
    return dp[target]
```

### Explanation

1. We first check if the total sum is odd (if so, equal partition is impossible).
2. We create a 1D DP array where `dp[i]` represents whether it's possible to achieve a sum of `i`.
3. We initialize `dp[0]` as True (empty subset has sum 0).
4. For each number, we update the DP array from right to left to avoid using the same number multiple times.
5. If `dp[target]` is True, we can partition the array equally.

Time complexity: O(n * target), Space complexity: O(target)

## Problem 16: Word Break

### Problem Statement
Given a string `s` and a dictionary of strings `wordDict`, return `true` if `s` can be segmented into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

### Solution

We can solve this using a 1D DP approach.

```{code-cell} python3
def wordBreak(s, wordDict):
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    word_set = set(wordDict)
    
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[n]
```

### Explanation

1. We use a 1D DP array where `dp[i]` represents whether the substring `s[:i]` can be segmented into dictionary words.
2. We initialize `dp[0]` as True (empty string is always valid).
3. For each index `i`, we check if any previous segmentation `dp[j]` is valid and if the substring `s[j:i]` is in the dictionary.
4. If both conditions are true, we mark `dp[i]` as True.
5. The final answer is in `dp[n]`.

Time complexity: O(n^2 * m), where m is the maximum length of a word in the dictionary.
Space complexity: O(n)

## Problem 17: Decode Ways

### Problem Statement
A message containing letters from A-Z can be encoded into numbers using the following mapping:

'A' -> "1"
'B' -> "2"
...
'Z' -> "26"

Given a string `s` containing only digits, return the number of ways to decode it.

### Solution

We can solve this using a 1D DP approach.

```{code-cell} python3
def numDecodings(s):
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    
    for i in range(2, n + 1):
        # Single digit
        if s[i-1] != '0':
            dp[i] += dp[i-1]
        
        # Two digits
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i-2]
    
    return dp[n]
```

### Explanation

1. We use a 1D DP array where `dp[i]` represents the number of ways to decode the first `i` characters.
2. We initialize base cases for empty string and first character.
3. For each position, we consider two cases:
   - If the current digit is not '0', we can decode it as a single character.
   - If the last two digits form a valid number (10-26), we can decode them together.
4. The sum of these two cases gives the number of ways to decode up to the current position.

Time complexity: O(n), Space complexity: O(n)

## Problem 18: Maximal Square

### Problem Statement
Given an `m x n` binary matrix filled with 0's and 1's, find the largest square submatrix of all 1's and return its area.

### Solution

We can solve this using a 2D DP approach.

```{code-cell} python3
def maximalSquare(matrix):
    if not matrix:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_side = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if matrix[i-1][j-1] == '1':
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                max_side = max(max_side, dp[i][j])
    
    return max_side * max_side
```

### Explanation

1. We use a 2D DP table where `dp[i][j]` represents the side length of the largest square ending at position (i-1, j-1) in the original matrix.
2. For each '1' in the matrix, we take the minimum of the three adjacent squares (left, top, top-left) and add 1.
3. We keep track of the maximum side length encountered.
4. The area of the largest square is the square of the maximum side length.

Time complexity: O(mn), Space complexity: O(mn)

## Problem 19: Longest Increasing Path in a Matrix

### Problem Statement
Given an `m x n` integers matrix, return the length of the longest increasing path in matrix. From each cell, you can either move in four directions: left, right, up, or down. You may not move diagonally or move outside the boundary (i.e., wrap-around is not allowed).

### Solution

We can solve this using DFS with memoization, which is a form of top-down DP.

```{code-cell} python3
def longestIncreasingPath(matrix):
    if not matrix:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    memo = [[0] * n for _ in range(m)]
    
    def dfs(i, j):
        if memo[i][j] != 0:
            return memo[i][j]
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        max_length = 1
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and matrix[ni][nj] > matrix[i][j]:
                max_length = max(max_length, 1 + dfs(ni, nj))
        
        memo[i][j] = max_length
        return max_length
    
    return max(dfs(i, j) for i in range(m) for j in range(n))
```

### Explanation

1. We use a memoization table to store the longest path starting from each cell.
2. For each cell, we perform a DFS to explore all possible increasing paths.
3. We use memoization to avoid redundant computations.
4. The maximum value in the memoization table gives the length of the longest increasing path.

Time complexity: O(mn), Space complexity: O(mn)

## Problem 20: Arithmetic Slices

### Problem Statement
An integer array is called arithmetic if it consists of at least three elements and if the difference between any two consecutive elements is the same. Given an integer array `nums`, return the number of arithmetic subarrays of `nums`.

### Solution

We can solve this using a 1D DP approach.

```{code-cell} python3
def numberOfArithmeticSlices(nums):
    n = len(nums)
    if n < 3:
        return 0
    
    dp = [0] * n
    total = 0
    
    for i in range(2, n):
        if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
            dp[i] = dp[i-1] + 1
            total += dp[i]
    
    return total
```

### Explanation

1. We use a 1D DP array where `dp[i]` represents the number of arithmetic slices ending at index `i`.
2. If three consecutive elements form an arithmetic sequence, we add 1