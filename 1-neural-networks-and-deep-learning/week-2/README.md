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

### Computation Graph Introduction 

Motivation 

- Suppose we want to compute a function $J(a,b,c) = 3(a+bc)$
  - This can be broken down in into smaller equations
    - $u = bc$
    - $v = a + u$
    - $J = 3v$
- A computation graph illustrates the manner in which we start with the input variables, $a,b,c$, and progress to the answer

Computation Graph 

- Illustrates the left-to-right (input-to-output) relationships necessary to compute a value 
  - One step of forward propagation on a computation graph yields a cost calculation
  - One step of backward propagation on a computation graph yields a derivative of the final output variable 

### Derivatives via Computation Graphs 

Code Convention 

- Typically there is a final output variable (FOV) that we care about (want to optimize)
  - We will need to calculate the derivative of the FOV wrt other variables 
    - Python 
      - `dvar`
        - represents dFOV/dvar

### Logistic Regression Gradient Descent 

Equation Recap 

- $z = w^Tx + b$
- $\hat{y} = a = σ(z)$
- $ℒ(a,y) = -(yloga + (1-y)log(1-a))$

### Logistic Regression Gradient Descent on $m$ Examples

Equation Recap 

- Cost Function 
  - $J(w,b) = \frac{1}{m} ∑ℒ(a^i,y)$
  - $a^i = \hat{y}^i = σ(z^i) = σ(w^Tx^i + b)$

Initialize
  
- $J=0$
- $dw_1=0$
- $dw_2=0$
- $db=0$

Iterate over $m$ training examples

- For $i=1$ to $m$
  - $z^i = w^Tx^i + b$
  - $a^i = σ(z^i)$
  - $J += -[y^iloga^i + (1-y^i)log(1-a^i)]$
  - $dz^i = a^i - y^i$
  - $dw_1 += x{_1}{^i}dz^i$
  - $dw_2 += x{_2}{^i}dz^i$
    - Two features $w_1$ and $w_2$
  - $db += dz^i$

Divide by $m$ 

- $J /= m$
- $dw_1 /= m$
- $dw_2 /= m$
- $db /= m$

Update features to complete one step of gradient descent

- $w_1 := w_1 - αdw_1$
- $w_2 := w_2 - αdw_2$
- $b := b - αdb$

Efficiency Note 

- This implementation would be incredibly inefficient 
- We need to vectorize the implementation

### Vectorization Introduction 

Vectorization 

- The act of removing explicit for loops in  your code 

Non-Vectorized

```python 
z = 0 
for i in range(n-x):
  z += w[i] * x[i]
z += b
```

Vectorized 

```python 
z = np.dot(w,x) + b
```

### Vectorizing Logistic Regression 

Matrix Variables 

- Training Matrix 
  - We defined the training matrix $X∈ℜ^{n_x×m}$ to consist of columns for each training example 
- $Z$ Matrix 
  - We can construct a matrix of $Z∈ℜ^{1×m}$ such that $Z=w^TX + B$, where $B∈ℜ^{1×m}$ containing the bias value 
  - Python 
    - `Z = np.dot(w.T, X) + b`
      - N.B. b is a 1-by-1 and accommodated via broadcasting
- Sigmoid Output Matrix $A$
  - We construct a matrix $A∈ℜ^{1×m}$ to contain the results of applying the sigmoid function to $Z$
- Ground Truth Matrix $Y$
  - We construct a matrix $Y∈ℜ^{1×m}$ to contain the labeled, ground truth data correlating to $y^1, ..., y^m$

### Vectorizing Logistic Regressions Gradient Output 

Derivative Matrix Variables 

- $dZ$ Matrix 
  - We define a matrix containing the $dz^1, ..., dz^m$ scalar values such that $dZ∈ℜ^{1×m}$
  - We can compute $dZ$ as $dZ=A-Y$
- $dw$ Vector 
  - Constructed as $dw = \frac{1}{m}XdZ^T$
  - Python 
    - `dw = np.dot(X, dZ.T) / m`
    - `dw = X @ dZ.T / m`

Vectorized Scalar Variables 

- $db$
  - As we have constructed $dZ$, we can use its sum in the calculation of $db$
  - Python 
    - `db = np.sum(dZ) / m`

### Vectorizing Logistic Regression Complete 

One Iteration of Gradient Descent 

- Python 

```python 
Z = np.dot(w.T, X) + b 
A = sigm(Z)
dZ = A - Y
dw = np.dot(X, dZ.T) / m 
db = np.sum(dZ) / m 

w = w - alpha * dw 
b = b - alpha
```

### Broadcasting in Python 

Example

- Consider a table with the calories from Carbs, Protein, and Fat in 100g of different foods

|         | Apples | Beef  | Eggs | Potatoes |
|---------|--------|-------|------|----------|
| Carb    | 56.0   | 0.0   | 4.4  | 68.0     |
| Protein | 1.2    | 104.0 | 52.0 | 8.0      |
| Fat     | 1.8    | 135.0 | 99.0 | 0.9      |

- We wish to calculate the percentage of calories from each type
- Solution
  - Let the table be a $3×4$ matrix, `A`

```python 
A = np.array([[56.0, 0.0, 4.4, 68.0], [1.2, 104.0, 52.0, 8.0], [1.8, 135.0, 99.0, 0.9]])
cals = A.sum(axis=0)
percents = 100 * A / cals
```

Broadcasting 

- Given an $m×n$ matrix, using $+,-,*,/$ on a $1×n$ matrix, the result will be $m×n$
- Given an $m×n$ matrix, using $+,-,*,/$ on a $m×1$ matrix, the result will be $m×n$
- Given a vector, using $+,-,*,/$ on a scalar, the result will be the same shape as the vector
