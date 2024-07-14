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

(chapter_8)=

# Chapter 8: Dynamic Programming in Arrays and Matrices

Dynamic Programming (DP) is particularly useful for solving problems involving arrays and matrices. In this chapter, we'll explore two classic problems: Matrix Chain Multiplication and Maximal Square.

## 8.1 Matrix Chain Multiplication

### Problem Statement

Given a sequence of matrices, find the most efficient way to multiply these matrices together. The problem is not to actually perform the multiplications, but merely to decide in which order to perform the multiplications.

For example, suppose you have three matrices A, B, and C with dimensions 10x30, 30x5, and 5x60 respectively. There are two ways to multiply them:

1. (AB)C = (10x30x5) + (10x5x60) = 1500 + 3000 = 4500 operations
2. A(BC) = (30x5x60) + (10x30x60) = 9000 + 18000 = 27000 operations

Clearly, the first way is more efficient.

### Approach

We can solve this problem using a 2D DP table. Let $dp[i][j]$ represent the minimum number of scalar multiplications needed to compute the matrix product from the $i$-th matrix to the $j$-th matrix.

The recurrence relation is:

$dp[i][j] = \min_{k=i}^{j-1} (dp[i][k] + dp[k+1][j] + d[i-1] * d[k] * d[j])$

where $d[i-1]$, $d[k]$, and $d[j]$ are the dimensions of the matrices.

### Implementation

```{code-cell} python3
def matrix_chain_multiplication(dimensions):
    n = len(dimensions) - 1  # number of matrices
    dp = [[0] * n for _ in range(n)]
    
    # len is chain length
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dimensions[i]*dimensions[k+1]*dimensions[j+1]
                dp[i][j] = min(dp[i][j], cost)
    
    return dp[0][n-1]

# Test the function
dimensions = [10, 30, 5, 60]
print(f"Minimum number of multiplications: {matrix_chain_multiplication(dimensions)}")
```

### Complexity Analysis

- Time Complexity: $O(n^3)$, where $n$ is the number of matrices.
- Space Complexity: $O(n^2)$ to store the DP table.

### Visualization

Here's a text-based visualization of how the DP table would be filled for the matrices with dimensions [10, 30, 5, 60]:

```
      0    1    2
0     0  1500 4500
1     0    0  9000
2     0    0     0
```

## 8.2 Maximal Square

### Problem Statement

Given an $m \times n$ binary matrix filled with 0's and 1's, find the largest square submatrix of 1's and return its area.

For example, given the matrix:
```
1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
```
The largest square submatrix of 1's has a size of 2x2, so the function should return 4.

### Approach

We can solve this problem using a 2D DP table. Let $dp[i][j]$ represent the side length of the largest square submatrix whose bottom right corner is at position (i, j) in the original matrix.

The recurrence relation is:

$dp[i][j] = \min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1$ if $matrix[i][j] == 1$

$dp[i][j] = 0$ if $matrix[i][j] == 0$

### Implementation

```{code-cell} python3
def maximal_square(matrix):
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

# Test the function
matrix = [
    ["1","0","1","0","0"],
    ["1","0","1","1","1"],
    ["1","1","1","1","1"],
    ["1","0","0","1","0"]
]
print(f"Area of the largest square submatrix: {maximal_square(matrix)}")
```

### Complexity Analysis

- Time Complexity: $O(mn)$, where $m$ and $n$ are the dimensions of the matrix.
- Space Complexity: $O(mn)$ to store the DP table.

### Visualization

Here's a text-based visualization of how the DP table would be filled for the given matrix:

```
0 0 0 0 0 0
0 1 0 1 0 0
0 1 0 1 1 1
0 1 1 1 2 2
0 1 0 0 1 0
```

The largest value in this DP table is 2, which corresponds to a 2x2 square, hence an area of 4.

## Conclusion

Dynamic Programming in arrays and matrices often involves 2D DP tables and can solve complex optimization problems efficiently. The key is to define the right recurrence relation and build the DP table step by step.

In the Matrix Chain Multiplication problem, we saw how DP can be used to optimize the order of operations. In the Maximal Square problem, we used DP to efficiently find patterns in a 2D grid.

These techniques can be applied to a wide range of problems involving sequences and grids, from image processing to optimizing computations.