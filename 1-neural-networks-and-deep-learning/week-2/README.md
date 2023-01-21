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
  - $(x^1,y^1), (x^2,y^2), ..., (x^m,y^m),$
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
