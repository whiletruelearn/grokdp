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

(chapter_5)=

# Advanced Dynamic Programming Techniques

In this chapter, we'll explore advanced Dynamic Programming techniques that are often used to solve more complex problems. We'll cover state compression, bitmasking in DP, and DP on trees.

## State Compression

State compression is a technique used to reduce the state space in DP problems, often by encoding the state into a single integer or using bitwise operations.

### Problem: Traveling Salesman Problem (TSP)

#### Problem Statement:
Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?

#### Solution using State Compression:

```{code-cell} python3
    import itertools

    def tsp(distances):
        n = len(distances)
        all_sets = [()]  # start with empty tuple
        for k in range(1, n):
            all_sets.extend(itertools.combinations(range(1, n), k))
        
        memo = {}
        
        def dp(pos, visited):
            if visited == all_sets[-1]:  # all cities have been visited
                return distances[pos][0]  # return to starting city
            
            state = (pos, visited)
            if state in memo:
                return memo[state]
            
            ans = float('inf')
            for next_city in range(1, n):
                if next_city not in visited:
                    new_visited = tuple(sorted(visited + (next_city,)))
                    ans = min(ans, distances[pos][next_city] + dp(next_city, new_visited))
            
            memo[state] = ans
            return ans
        
        return dp(0, ())

    # Example usage
    distances = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    print(tsp(distances))  # Output: 80
```

#### Explanation:
- We use tuples to represent the set of visited cities, which allows for efficient hashing and memoization.
- The state is compressed into a tuple `(current_city, visited_cities)`.
- We use memoization to avoid redundant calculations.
- The time complexity is $O(n^2 2^n)$, which is a significant improvement over the naive $O(n!)$ approach.

## Bitmasking in DP

Bitmasking is a technique where we use bits to represent a subset of elements. It's particularly useful in DP problems involving subsets.

### Problem: Subset Sum

#### Problem Statement:
Given an array of integers and a target sum, determine if there exists a subset of the array that adds up to the target sum.

#### Solution using Bitmasking:

```{code-cell} python3
    def subset_sum(nums, target):
        n = len(nums)
        dp = [False] * (1 << n)
        dp[0] = True  # empty subset sums to 0
        
        for mask in range(1, 1 << n):
            for i in range(n):
                if mask & (1 << i):
                    prev_mask = mask ^ (1 << i)
                    if dp[prev_mask]:
                        dp[mask] = dp[prev_mask] or (sum(nums[j] for j in range(n) if mask & (1 << j)) == target)
        
        return dp[-1]

    # Example usage
    nums = [3, 34, 4, 12, 5, 2]
    target = 9
    print(subset_sum(nums, target))  # Output: True
```


#### Explanation:
- We use a bit mask to represent subsets. For example, 101 in binary represents the subset containing the first and third elements.
- `dp[mask]` is True if there exists a subset represented by `mask` that sums to the target.
- We iterate through all possible subsets (2^n) and build up the solution.
- The time complexity is $O(n2^n)$, which is efficient for small to medium-sized inputs.

## DP on Trees

DP can also be applied to tree structures, often using a post-order traversal to build solutions from the leaves up to the root.

### Problem: Binary Tree Maximum Path Sum

#### Problem Statement:
Given a binary tree, find the maximum path sum. The path may start and end at any node in the tree.

#### Solution:

```{code-cell} python3
    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def max_path_sum(root):
        def max_gain(node):
            nonlocal max_sum
            if not node:
                return 0
            
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)
            
            current_max_path = node.val + left_gain + right_gain
            max_sum = max(max_sum, current_max_path)
            
            return node.val + max(left_gain, right_gain)
        
        max_sum = float('-inf')
        max_gain(root)
        return max_sum

    # Example usage
    root = TreeNode(10)
    root.left = TreeNode(2)
    root.right = TreeNode(10)
    root.left.left = TreeNode(20)
    root.left.right = TreeNode(1)
    root.right.right = TreeNode(-25)
    root.right.right.left = TreeNode(3)
    root.right.right.right = TreeNode(4)
    print(max_path_sum(root))  # Output: 42
```

#### Explanation:
- We use a post-order traversal to compute the maximum path sum.
- For each node, we calculate the maximum gain that can be obtained by including that node in a path.
- We update the global maximum sum at each node by considering the path that goes through the node and potentially its left and right subtrees.
- The time complexity is O(n), where n is the number of nodes in the tree.

These advanced DP techniques allow us to solve complex problems more efficiently. State compression and bitmasking help us handle problems with large state spaces, while DP on trees extends our DP skills to hierarchical structures.

In the next chapter, we'll explore some classic DP problems that often appear in coding interviews and discuss strategies for approaching them.