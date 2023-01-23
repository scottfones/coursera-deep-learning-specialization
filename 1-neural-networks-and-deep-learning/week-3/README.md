# Neural Networks and Deep Learning: Week 3 - Notes

## Shallow Neural Networks 

### Neural Networks Overview

Notation

- Compared to the single node of a Logistic Regression, we will now combine nodes into layers.
  - We need to be able to distinguish between a layer and an example
    - Layer 
      - A superscripted number inside brackets will indicate a layer designation
      - $z^{[1]} = W^{[1]}x + b^{[1]}$
        - $z$, $W$, and $b$ belong to the first layer of the network 
    - Example 
      - A superscripted number inside parenthesis will indicate a training example 
      - $x^{(1)}$
        - The first training example 

### Neural Network Representation

Diagram 

- Input Layer
  - The input layer consists of a vector of input data 
  - $a^{[0]} = x$
    - The $a$ refers to the activations of the input layer 
      - The components of $x$ that pass data into the hidden layer 
- Hidden Layer 
  - The intermediate, hidden layers of the network 
  - $a^{[1]}$
    - $a^{[1]}$ is a $m×1$ matrix where each row represents a node in the layer
    - The components of $a^{[1]}$ are identified with notation $a{^{[1]}}{_i}$, where $i$ is the specific node in the layer
  - Each layer is associated with values of $w$ and $b$
    - $a^{[1]}$ is associated with $w^{[1]}$ and $b^{[1]}$
      - $w^{[1]}$ will be $m×n$ where $m$ is the number of nodes in the layer and $n$ is the number of inputs
      - $b^{[1]}$ will be $m×1$ where $m$ is the number of nodes in the layer
- Output Layer 
  - The layer responsible for outputting $\hat{y}$
  - $a^{[2]}$
  - $\hat{y} = a^{[2]}$


