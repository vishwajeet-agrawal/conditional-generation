Our model is given as
$$P(X_i | X_S) \propto \exp(\phi_i(X_i)^T \psi_{i, S}(X_S))$$

We first experiment for the binary variable case when $X \in \{0, 1\}^n$.

Then we define learnable embedding $\phi_i(X_i) = z_{i,v} \in \mathbb{R}^d$ with $v = X_i$ for each $i \in [n]$ and $X_i \in \{0, 1\}$, and $d$ is the embedding dimension.

We define $\psi_{i, S}(X_S)$ as
$$aggregate(\{(i, j, \phi'(X_j))\}_{j\in S})$$
or letting $i, j$ represented as a vectors $b_i, b_j \in \mathbb{R}^d$ and $\psi'(X_j)$ as $z_{j, v}$ for $v = X_j$.

Ways to implement $aggregate$ function

> Average
$$\psi_{i, S}(X_S) = 1/|S|\sum_{j\in S} (b_j + b_i + z_{j,X_j}) $$
>Attention

$$w'_{jk} = (b_j + b_i + z_{j, X_j})^T(b_k+b_i+z_{k, X_k})$$
$$w_{jk} = \exp(w'_{jk}/\sqrt{d})/\sum_{k\in S}\exp(w'_{jk}/\sqrt{d})$$
$$\psi_{i, S}(X_S) = 1/|S|\sum_{j\in S}\sum_{k\in S} w_{jk}(b_k + b_i + z_{k, X_k})$$


> Next Steps

Multiple rounds of message passing, either through attention or MLP.

![View Image](tv_dist_vs_embedding_dim.png)
