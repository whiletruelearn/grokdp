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

(chapter_12)=

# Common Patterns and Problem-Solving Strategies

Dynamic Programming (DP) problems often share common patterns and can be approached with similar strategies. In this chapter, we'll explore how to identify DP problems, common patterns in DP, and a step-by-step approach to solving them.

## 12.1 Identifying DP Problems

DP is typically applicable when a problem has the following characteristics:

1. **Optimal Substructure**: The optimal solution to the problem can be constructed from optimal solutions of its subproblems.
2. **Overlapping Subproblems**: The problem can be broken down into subproblems which are reused several times.

Common indicators that a problem might be solvable using DP include:

- The problem asks for the maximum, minimum, or optimal value
- The problem involves making choices to arrive at a solution
- The problem asks for the number of ways to do something
- The problem involves sequences or series

## 12.2 Common DP Patterns

Here are some common patterns you'll encounter in DP problems:

### 1. Linear Sequence DP

Example problems: Fibonacci sequence, Climbing Stairs

Pattern:
```python3
dp[i] = some_function(dp[i-1], dp[i-2], ...)
```

### 2. Two-Dimensional DP

Example problems: Longest Common Subsequence, Edit Distance

Pattern:
```python3
dp[i][j] = some_function(dp[i-1][j], dp[i][j-1], dp[i-1][j-1], ...)
```

### 3. Interval DP

Example problems: Matrix Chain Multiplication, Optimal Binary Search Tree

Pattern:
```python3
for length in range(2, n+1):
    for i in range(n-length+1):
        j = i + length - 1
        dp[i][j] = some_function(dp[i][k], dp[k+1][j], ...)
```

### 4. Subset DP

Example problems: Subset Sum, Partition Equal Subset Sum

Pattern:
```python3
dp[i][s] = dp[i-1][s] or dp[i-1][s-nums[i]]
```

### 5. String DP

Example problems: Longest Palindromic Subsequence, Regular Expression Matching

Pattern:
```python3
dp[i][j] = some_function(dp[i+1][j-1], s[i], s[j], ...)
```

## 12.3 Steps to Approach and Solve DP Problems

Follow these steps to solve DP problems:

1. **Identify the problem as DP**: Look for optimal substructure and overlapping subproblems.

2. **Define the state**: Determine what information you need to represent a subproblem.

3. **Formulate the recurrence relation**: Express the solution to a problem in terms of solutions to smaller subproblems.

4. **Identify the base cases**: Determine the simplest subproblems and their solutions.

5. **Decide on the implementation approach**: Choose between top-down (memoization) or bottom-up (tabulation) approach.

6. **Implement the solution**: Write the code, ensuring all subproblems are solved before they are needed.

7. **Optimize if necessary**: Look for opportunities to optimize space or time complexity.

Let's apply these steps to a classic DP problem: the Longest Increasing Subsequence (LIS).

```{code-cell} python3
def longest_increasing_subsequence(nums):
    if not nums:
        return 0
    
    n = len(nums)
    # Step 2: Define the state
    # dp[i] represents the length of the LIS ending at index i
    dp = [1] * n
    
    # Step 3: Formulate the recurrence relation
    # Step 4: Identify the base cases (implicit in the initialization of dp)
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    # Return the maximum value in dp
    return max(dp)

# Test the function
nums = [10, 9, 2, 5, 3, 7, 101, 18]
print(f"Length of Longest Increasing Subsequence: {longest_increasing_subsequence(nums)}")
```

## 12.4 Tips for Coding Interviews

1. **Start with a brute force solution**: Even if it's not efficient, it helps understand the problem and might give insights into the optimal solution.

2. **Draw out examples**: Visualizing the problem can help identify patterns and the structure of the solution.

3. **Think about the state**: What information do you need to solve a subproblem?

4. **Consider different DP patterns**: Try to map the problem to one of the common patterns we discussed.

5. **Optimize after solving**: First get a working solution, then think about optimizations.

6. **Practice, practice, practice**: Familiarity with common DP problems will help you recognize patterns quickly.

## Conclusion

Dynamic Programming is a powerful technique, but it requires practice to master. By understanding common patterns and following a structured approach, you can become proficient at recognizing and solving DP problems.

Remember, the key to DP is breaking down a complex problem into simpler subproblems and storing the results for reuse. With the strategies and patterns we've discussed in this chapter, you're well-equipped to tackle a wide range of DP problems in coding interviews and real-world applications.

In the next chapter, we'll put these strategies into practice by solving a collection of DP problems from actual coding interviews.