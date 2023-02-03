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
