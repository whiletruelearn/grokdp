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

(chapter_1)=

# Introduction to Dynamic Programming

## What is Dynamic Programming?

Dynamic Programming (DP) is a powerful problem-solving technique used in computer science and mathematics. It's particularly useful for solving optimization problems and certain types of coding challenges. The core idea behind dynamic programming is to break down a complex problem into simpler subproblems, solve these subproblems only once, and store their solutions for future use.

The term "dynamic programming" was coined by Richard Bellman in the 1950s. Despite its name, it has nothing to do with dynamic programming languages or dynamic memory allocation. Instead, "programming" in this context refers to a tabular method of solving problems.

## When to use Dynamic Programming

Dynamic Programming is particularly useful when:

1. The problem has overlapping subproblems
2. The problem has an optimal substructure
3. The problem involves finding an optimum value (maximum or minimum)

Let's break these down:

### 1. Overlapping Subproblems

A problem has overlapping subproblems if it can be broken down into subproblems which are reused several times. 

For example, consider the Fibonacci sequence:
    
```{code-cell} python3
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
  ```
In this recursive implementation, `fibonacci(5)` will calculate `fibonacci(3)` twice. As `n` grows, the number of redundant calculations increases exponentially.

### 2. Optimal Substructure

A problem has optimal substructure if an optimal solution can be constructed from optimal solutions of its subproblems.

For instance, the shortest path problem has an optimal substructure: if the shortest path from A to C goes through B, then the path from A to B and the path from B to C must also be the shortest paths between those points.

### 3. Finding an Optimum Value

Many DP problems involve finding a maximum or minimum value. Examples include:

- Finding the longest common subsequence
- Minimizing the number of coins to make change
- Maximizing the value of items in a knapsack

## Key Concepts: Overlapping Subproblems and Optimal Substructure

Let's dive deeper into these two fundamental concepts of Dynamic Programming.

### Overlapping Subproblems

When a problem has overlapping subproblems, it means that the same subproblems are solved multiple times. This is where DP shines - by storing the results of these subproblems, we can avoid redundant computation.

Consider this simple diagram of the Fibonacci calculation for `n=5`:

                 fib(5)
               /        \
          fib(4)         fib(3)
         /      \       /      \
      fib(3)   fib(2) fib(2)  fib(1)
      /    \
    fib(2) fib(1)

Notice how `fib(3)` and `fib(2)` appear multiple times. In DP, we would calculate these only once and reuse the results.

### Optimal Substructure

A problem has optimal substructure if its overall optimal solution can be constructed from the optimal solutions of its subproblems.

Let's consider the shortest path problem as an example:

        A --- 3 --- B --- 2 --- C
         \         /
          \       /
           5     4
            \   /
             \ /
              D

If the shortest path from A to C is A-B-C (with a total distance of 5), then:
- A-B must be the shortest path from A to B
- B-C must be the shortest path from B to C

This property allows us to build the solution to the larger problem (shortest path from A to C) from solutions to smaller subproblems (shortest paths A-B and B-C).

In the next chapters, we'll explore how to leverage these properties to solve complex problems efficiently using Dynamic Programming techniques.