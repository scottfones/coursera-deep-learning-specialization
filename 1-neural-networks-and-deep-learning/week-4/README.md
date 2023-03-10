# Neural Networks and Deep Learning: Week 4 - Notes

## Deep Neural Networks 

### Deep L-Layer Network

Shallow vs Deep 

- Logistic regression is a shallow neural network 
  - Single layer 
- What qualifies as deep isn't concrete
- Deep networks can learn functions that shallow networks can't

### Notation 

Layers 

- Uppercase $L$
  - $L=4$
    - A network with 4 layers 

Layer Units

- An $n$ with a superscript, lowercase $l$, $n^{[l]}$
  - $n^{[l]} = 5$
    - There are 5 units (nodes) in layer $l$

Layer Activations 

- An $a$ with a superscript, lowercase $l$, $a^{[l]}$
  - $a^{[l]} = g^{[l]}(z^{[l]})$
    - The activation vector for layer $l$ 
- N.B. 
  - $x = a^{[0]}$
  - $a^{[L]} = \hat{y}$

Layer Weights 

- A $W$ with a superscript, lowercase $l$, $W^{[l]}$

Layer Bias 

- A $b$ with a superscript, lowercase $l$, $b^{[l]}$

### Forward Propagation in a Deep Network 

Single Training Example

- Given a single input $x$
  - Layer 1 
    - $z^{[1]} = W^{[1]}x + b^{[1]}$ 
    - $a^{[1]} = g^{[1]}(z^{[1]})$
  - Layer 2
    - $z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$ 
    - $a^{[2]} = g^{[2]}(z^{[2]})$
  - Output Layer
    - $z^{[L]} = W^{[L]}a^{[L-1]} + b^{[L]}$ 
    - $a^{[L]} = \hat{y} = g^{[L]}(z^{[L]})$
  - General Equation for Layer $l$
    - $z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$ 
    - $a^{[l]} = g^{[l]}(z^{[l]})$
    
Vectorized Example

- Given a matrix of training input $X$
  - Layer 1 
    - $Z^{[1]} = W^{[1]}X + b^{[1]}$ 
      - $X = A^{[0]}$
    - $A^{[1]} = g^{[1]}(Z^{[1]})$
  - Layer 2 
    - $Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$ 
    - $A^{[2]} = g^{[2]}(Z^{[2]})$
  - Output Layer
    - $Z^{[L]} = W^{[L]}a^{[L-1]} + b^{[L]}$ 
    - $A^{[L]} = \hat{Y} = g^{[L]}(z^{[L]})$

### Getting Matrix Dimensions Right 

Dimensions 

$W^{[l]}$
  - $W^{[l]} = (n^{[l]} × n^{[l-1]}$
$b^{[l]}$
  - $b^{[l]} = (n^{[l]} × 1)$
$a^{[l]}$ and $z^{[l]}$
  - $a^{[l]} = z^{[l]} = (n^{[l]} × 1)$
$A^{[l]}$ and $Z^{[l]}$
  - $A^{[l]} = Z^{[l]} = (n^{[l]} × m)$
    - If $l = 0$
      - $A^{[0]} = X = Z^{[l]} = (n^{[0]} × m)$
$dW^{[l]}$
  - $dW^{[l]} = (n^{[l]} × n^{[l-1]}$
$db^{[l]}$
  - $db^{[l]} = (n^{[l]} × 1)$
$dA^{[l]}$ and $dZ^{[l]}$
  - $A^{[l]} = Z^{[l]} = (n^{[l]} × m)$


