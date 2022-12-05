# Group Theory 101

# Representation Theory 101

# Previous Work

# Representations of the cyclic group


\text{tr}(\rho(x)\rho(y)\rho(z^{-1}))=2\cos(w(x+y-z))


# Proposed Algorithm

Take an injective group representation $\rho : G \to GL(n)$


Map $x, y \to \rho(x), \rho(y)$

  

For each output logit $z$ map this to $\text{tr}(\rho(x)\rho(y)\rho(z^{-1})) = \text{tr}(\rho(xyz^{-1}))$.

  

Theorem: $\text{tr}(\rho(g))\leq n$ with equality iff $\rho(g)=I$ 

Sketch proof: This holds because trace is the sum of the eigenvalues, and $\rho(g)^{|G|}=I$, so the eigenvalues are nth roots of unity, and this holds by triangle inequality.

  

So $\arg \max_z(\text{tr}(\rho(xyz^{-1}))) = xy$, and if $\rho$ is injective this is unique, and if we scale up the logits enough, softmax limits to an argmax, so for any representation $\rho$ the largest logit is $z = xy$.