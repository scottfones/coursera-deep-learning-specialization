# Neural Networks and Deep Learning: Week 2 - Notes

## Binary Classification (Logistic Regression)

Typical Problem

- Input: Image
- Output: 1 or 0 (cat or not cat)

Image Breakdown

- Given an m-by-n image 
  - An m-by-n matrix for each of the RGB channels 
- Convert to Input 
  - Create a feature vector, $x$ by unrolling the matrices into one vector 
    - Given a 64-by-64 image, the feature vector is 12288-by-1 
      - $n=n_x=12288$

Binary Classifier

- Takes a feature vector, $x$, and predicts whether the corresponding label, $y$, is $1$ or $0$
  - In the above example, it takes an image and determines whether it is a cat or not.

### Notation

Training Example 

- A single training example is represented by a pair, $(x,y)$, where $x∈ℝ^{n_x}, y∈{0,1}$
- A set of $m$ training examples:
  - $(x^1,y^1), (x^2,y^2), ..., (x^m,y^m)$
  - Lowercase $m$ will be used to denote the number of training examples 
    - $m = m_{train}$
  - $m_{test}$ denotes the number of test examples
- Compact Notation 
  - Input Matrix
    - Define an input matrix, $X$, where the columns are the input values
      - $X=[x^1 x^2 ... x^m]$
      - Dimensions
        - $n_x$ rows
        - $m$ columns
        - $X∈ℝ^{n_x × m}$
        - `X.shape = (nx, m)`
  - Output Matrix 
    - Define an output matrix, $Y$, where each element $y∈{0, 1}$
      - $Y=[y^1 y^2 ... y^m]$
      - Dimensions 
        - 1 row 
        - $m$ columns 
        - $Y∈ℝ^{1×m}$
        - `Y.shape = (1, m)`

### Logistic Regression 

Definition 

- Given $x$, we want $\hat{y}=P(y=1 | x)$
  - $x∈ℜ^{n_x}$
  - $0 ≤ \hat{y} ≤ 1$

Regression Parameters

- $w∈ℜ^{n_x}$
- $b∈ℜ$

Output 

- $\hat{y} = σ(w^Tx + b) = σ(z)$
  - where 
    - $w^Tx + b = z$ 
    - $σ$ is the sigmoid function
      - $σ(z) = (1 + e^{-z})^{-1}$
      - $0 ≤ σ(z) ≤ 1$ and $σ(0) = 0.5$

### Logistic Regression Cost function

Problem 

- Given a training set, $\{(x^1, y^1), ..., (x^m, y^m)\}$, we want to find $\hat{y}^i ≈ y^i$
  - We want to find values for $w$ and $b$ such that our predictions, $\hat{y}^i$, will be close to our ground truth labels, $y^i$

Loss (Error) Function 

- We seek a measure for how good our output $\hat{y}$ is when the true label is $y$
  - Avoid L2-norm (squared error) as the result is non-convex for logistic regression and this interferes with gradient descent
- Definition 
  - $ℒ(\hat{y}, y) = -(ylog\hat{y} + (1-y)log(1-\hat{y}))$
- Intuition 
  - We want to minimize the error. Consider the boundary conditions,
    - If $y=1$: $ℒ(\hat{y}, y) = -log\hat{y}$
      - We want $-log\hat{y}$ to be as small as possible
        - Requires $log\hat{y}$ be large ∴ $\hat{y}$ must be large
    - If $y=0$: $ℒ(\hat{y}, y) = -log(1-\hat{y})$
      - We want $-log(1-\hat{y})$ to be as small as possible
        - Requires $log(1-\hat{y})$ be large ∴ $\hat{y}$ must be small

Cost Function 

- Measures how well we're doing over the entire training set 
- Definition 
  - $J(w,b) = \frac{1}{m} ∑ℒ(\hat{y}^i, y^i)$
  - $J(w,b) = -\frac{1}{m} ∑[y^ilog\hat{y}^i + (1-y^i)log(1-\hat{y}^i)]$

Convention

- Loss function applies only to a specific training example 
- Cost function is the cost of the parameters 
  - Looking for parameters $w$ and $b$ that minimize the cost function

### Gradient Descent Introduction

Motivation

- We want to find parameters $w$ and $b$ that minimize $J(w,b)$
  - As $J(w,b)$ is convex, we seek the minimum value on the surface of $J$

Method

- Assuming we are optimizing on $w$, we repeatedly update $w$ such that 
  - $w := w - α\frac{d}{dw}J(w)$
    - where 
      - $α$ is the learning rate
      - $\frac{d}{dw}J(w)$ is the change to make on $w$
  - Code Convention 
    - `w = w - a*dw`
- Optimizing $J(w,b)$
  - $w := w - α\frac{δ}{δw}J(w,b)$
  - $b := b - α\frac{δ}{δb}J(w,b)$


