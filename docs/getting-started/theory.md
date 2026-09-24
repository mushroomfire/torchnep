# NEP theory

This chapter describes the model TorchNEP trains — the neuroevolution potential (NEP) in its current form, NEP4 [[1–4]](#references) — and how TorchNEP fits it. None of it is needed to train or use a model: to get started, skip to the [User guide](../guide/training.md).

The formulas follow the NEP papers and GPUMD's documentation; where TorchNEP differs from GPUMD, which is only in how the parameters are fitted, the section says so.

## Energy

The potential energy of a structure is a sum of site energies, one per atom, plus an optional short-range repulsion:

$$
U = \sum_i U_i + U_\text{ZBL} .
$$

The site energy $U_i$ depends only on the neighbours of atom $i$ within a cutoff radius, through a **descriptor** vector $\mathbf q^i$ that is invariant under translations, rotations and permutations of like atoms. A neural network maps the descriptor to the energy.

## Neural network

Each element $t$ has its own network with one hidden layer of $N_\text{neu}$ neurons (`neuron` in `nep.in`):

$$
U_i = \sum_{\mu=1}^{N_\text{neu}} w^{(1)}_{t\mu} \tanh\!\Big(\sum_{\nu=1}^{N_\text{des}} w^{(0)}_{t\mu\nu}\, \tilde q^{\,i}_\nu - b^{(0)}_{t\mu}\Big) - b^{(1)} ,
\qquad t = \text{element of atom } i ,
$$

with the weights $w^{(0)}$, $w^{(1)}$ and thresholds $b^{(0)}$ of element $t$ and one energy offset $b^{(1)}$ shared by all elements. The input is the scaled descriptor $\tilde q^{\,i}_\nu = s_\nu q^i_\nu$ (see [Scaling](#scaling)).

## Descriptor

### Radial functions

Every neighbour $j$ of atom $i$ at distance $r_{ij}$ enters through radial functions built from Chebyshev polynomials $T_k$ and a smooth cutoff,

$$
f_\text{c}(r) = \begin{cases} \tfrac12 \big[1 + \cos(\pi r / r_\text{c})\big] , & r \le r_\text{c} , \\ 0 , & r > r_\text{c} , \end{cases}
\qquad
f_k(r) = \tfrac12 \Big[ T_k\big(2 (r/r_\text{c} - 1)^2 - 1\big) + 1 \Big] f_\text{c}(r) ,
$$

$k = 0, \dots, N_\text{bas}$, combined with trainable coefficients that depend on the elements $t_i$, $t_j$ of the pair:

$$
g_n(r_{ij}) = \sum_{k=0}^{N_\text{bas}} c^{\,t_i t_j}_{nk}\, f_k(r_{ij}) .
$$

The radial and the angular descriptor components use separate sets of these functions, each with its own cutoff, basis size and number of functions: $r_\text{c}^\text{R}$, $N_\text{bas}^\text{R}$, $n_\text{max}^\text{R}$ and $r_\text{c}^\text{A}$, $N_\text{bas}^\text{A}$, $n_\text{max}^\text{A}$ (`cutoff`, `basis_size`, `n_max` in `nep.in`). With per-species cutoffs (`cutoff rR1 rA1 rR2 rA2 ...`), a pair uses the mean of the two elements' values, $r_\text{c}^{\,t_i t_j} = \tfrac12 (r_\text{c}^{\,t_i} + r_\text{c}^{\,t_j})$.

### Radial components

$$
q^i_n = \sum_{j \ne i} g_n(r_{ij}) , \qquad 0 \le n \le n_\text{max}^\text{R} .
$$

### Angular components

The angular components start from the expansion of the neighbour density in spherical harmonics,

$$
A^i_{nlm} = \sum_{j \ne i} g_n(r_{ij})\, Y_{lm}(\hat{\mathbf r}_{ij}) .
$$

The **3-body** components contract two of them into a rotational invariant ($0 \le n \le n_\text{max}^\text{A}$, $1 \le l \le l_\text{max}^\text{3b}$):

$$
q^i_{nl} = \sum_{m=-l}^{l} (-1)^m A^i_{nlm} A^i_{nl(-m)}
= \frac{2l+1}{4\pi} \sum_{j \ne i} \sum_{k \ne i} g_n(r_{ij})\, g_n(r_{ik})\, P_l(\cos\theta_{ijk}) ,
$$

by the addition theorem of spherical harmonics, with $P_l$ the Legendre polynomials and $\theta_{ijk}$ the angle between $\mathbf r_{ij}$ and $\mathbf r_{ik}$. The left form costs one pass over the neighbours instead of a double sum; TorchNEP computes it with real spherical harmonics written as polynomials of the unit vector.

**Higher-body** components couple three or four $A^i_{nlm}$ of the same $n$ into rotational invariants. The flags after the first number of `l_max` switch them on, in GPUMD's order: $q^i_{n222}$ (4-body, three $l=2$ expansions), $q^i_{n1111}$ (5-body, four $l=1$), and the further 4-body couplings $q^i_{n112}$, $q^i_{n123}$, $q^i_{n233}$, $q^i_{n134}$. For example

$$
q^i_{n222} = \sum_{m_1 m_2 m_3} \begin{pmatrix} 2 & 2 & 2 \\ m_1 & m_2 & m_3 \end{pmatrix} A^i_{n2m_1} A^i_{n2m_2} A^i_{n2m_3}
$$

with the Wigner 3j symbol; see [[3]](#references) for the others. Each switched-on coupling adds $n_\text{max}^\text{A} + 1$ components. $q^i_{n1111}$ is a constant times $(q^i_{n1})^2$ and adds no information, which TorchNEP warns about.

### Scaling

The descriptor components differ in magnitude by orders, so each is scaled before the network, $\tilde q_\nu = s_\nu q_\nu$ with

$$
s_\nu = \frac{1}{\max_i q^i_\nu - \min_i q^i_\nu}
$$

over the atoms of the training set. The scale factors are computed once, from the descriptors of the initial model on the training data, and then kept fixed; they are stored in `nep.txt`. With `use_gpumd_qscaler=True` they are computed as GPUMD does, with all descriptor coefficients set to 1.

### Dimensions

| Number of | Count |
|---|---|
| radial components | $n_\text{max}^\text{R} + 1$ |
| 3-body components | $(n_\text{max}^\text{A} + 1)\, l_\text{max}^\text{3b}$ |
| higher-body components | $n_\text{max}^\text{A} + 1$ per switched-on coupling |
| descriptor size $N_\text{des}$ | sum of the above |
| descriptor coefficients $c^{\,t_i t_j}_{nk}$ | $N_\text{typ}^2 \big[(n_\text{max}^\text{R}+1)(N_\text{bas}^\text{R}+1) + (n_\text{max}^\text{A}+1)(N_\text{bas}^\text{A}+1)\big]$ |
| network parameters | $(N_\text{des} + 2)\, N_\text{neu}\, N_\text{typ} + 1$ |
| **trainable parameters in total** | $N_\text{typ}^2 \big[(n_\text{max}^\text{R}+1)(N_\text{bas}^\text{R}+1) + (n_\text{max}^\text{A}+1)(N_\text{bas}^\text{A}+1)\big] + (N_\text{des} + 2)\, N_\text{neu}\, N_\text{typ} + 1$ |

For example, a three-element model with `n_max 4 4`, `basis_size 8 8`, `l_max 4 2 0` and `neuron 30` has $N_\text{des} = 5 + 5 \cdot 4 + 5 = 30$ descriptor components, $9 \cdot (45 + 45) = 810$ descriptor coefficients and $32 \cdot 30 \cdot 3 + 1 = 2881$ network parameters, 3691 in total.

## Short-range repulsion (ZBL)

At short distances, where the training data are sparse, a screened Coulomb repulsion [[5, 6]](#references) keeps atoms apart:

$$
U_\text{ZBL} = \sum_{i<j} \frac{1}{4\pi\varepsilon_0} \frac{Z_i Z_j e^2}{r_{ij}}\, \phi\!\left(\frac{r_{ij}}{a_{ij}}\right) f_\text{s}(r_{ij}) ,
\qquad
a_{ij} = \frac{0.46848}{Z_i^{0.23} + Z_j^{0.23}}\ \text{Å} ,
$$

with the universal screening function

$$
\phi(x) = 0.18175\, e^{-3.1998 x} + 0.50986\, e^{-0.94229 x} + 0.28022\, e^{-0.4029 x} + 0.02817\, e^{-0.20162 x}
$$

and a switching function that is 1 below an inner radius $r_\text{in}$, 0 above an outer radius $r_\text{out}$ and $\tfrac12 \big[1 + \cos\big(\pi \frac{r - r_\text{in}}{r_\text{out} - r_\text{in}}\big)\big]$ between. The keywords of `nep.in` choose the radii:

- `zbl r`: $r_\text{out} = r$, $r_\text{in} = r/2$ for every pair;
- `zbl r` with `use_typewise_cutoff_zbl f`: $r_\text{out} = \min\big(f (R_i + R_j), r\big)$ from the covalent radii $R$, $r_\text{in} = 0$;
- `zbl zbl.in`: $r_\text{in}$, $r_\text{out}$ and the eight coefficients of $\phi$ per element pair, from GPUMD's `zbl.in` file.

The neural network is trained on top of $U_\text{ZBL}$: it learns the difference between the reference energy and the repulsion.

## Forces and virial

With $\mathbf r_{ij} = \mathbf r_j - \mathbf r_i$, the site energy $U_i$ depends on the positions only through the vectors $\mathbf r_{ij}$ to its neighbours. The force on atom $i$ and the virial follow from the partial derivatives $\partial U_i / \partial \mathbf r_{ij}$ [[1]](#references):

$$
\mathbf F_i = \sum_{j \ne i} \left( \frac{\partial U_i}{\partial \mathbf r_{ij}} - \frac{\partial U_j}{\partial \mathbf r_{ji}} \right) ,
\qquad
\mathbf W = -\sum_i \sum_{j \ne i} \mathbf r_{ij} \otimes \frac{\partial U_i}{\partial \mathbf r_{ij}} .
$$

TorchNEP computes $\partial U_i / \partial \mathbf r_{ij}$ analytically by default (the chain rule through the network, the descriptor and the radial functions), or by automatic differentiation with `use_autograd_forces=True`; both give the same forces.

## Training

TorchNEP fits the same model as GPUMD's `nep` executable, but differently. GPUMD minimises a sum of **root-mean-square** errors plus $\mathcal L_1$ and $\mathcal L_2$ penalties with the separable natural evolution strategy (SNES), a gradient-free method that gave NEP its name [[1]](#references). TorchNEP minimises a sum of **mean-square** errors by gradient descent, over mini-batches of structures:

$$
L = \frac{\lambda_e}{N_E} \sum_{n} w_n\, \Delta u_n^2
  + \frac{\lambda_f}{3 N_F} \sum_{n} w_n \sum_{i \in n} \lvert \Delta \mathbf F_i \rvert^2
  + \frac{\lambda_v}{6 N_V} \sum_{n} w_n\, \lvert \Delta \mathbf W_n \rvert^2 ,
$$

with $\Delta u_n$ the energy error per atom of structure $n$, $\Delta \mathbf F_i$ the force error of atom $i$ and $\Delta \mathbf W_n$ the six independent components of the virial per atom; $N_E$, $N_F$ and $N_V$ count the energy labels, force-labelled atoms and virial labels, and $w_n$ is the structure's [weight](training-data.md#frame-weights) (1 by default). The weights $\lambda_e$, $\lambda_f$, $\lambda_v$ are `lambda_e`, `lambda_f`, `lambda_v` in `nep.in`, and switch to the `stage2_lambda_*` values in the second stage of training.

- **Parameters.** The optimiser trains the descriptor coefficients $c^{\,t_i t_j}_{nk}$ and the network parameters $w^{(0)}$, $b^{(0)}$, $w^{(1)}$. The scale factors $s_\nu$ stay fixed.
- **Energy offset.** $b^{(1)}$ shifts every energy by the same amount, so it is solved exactly instead of learned: after each epoch, $b^{(1)} \leftarrow b^{(1)} + \sum_n w_n r_n / \sum_n w_n$ with $r_n$ the energy residual per atom — the offset that minimises the energy term for the current weights.
- **Optimiser.** AdamW [[7]](#references) (Adam with decoupled weight decay, AMSGrad variant [[8]](#references)); the weight decay (`weight_decay`) takes the place of GPUMD's $\mathcal L_2$ penalty and acts on all trained parameters. The learning rate decreases when the loss stops improving (`lr_scheduler plateau`) or in steps; gradients are clipped at `max_grad_norm`.
- **Two stages and model selection.** See [Training](../guide/training.md): the second stage changes the loss weights and restarts the learning rate, `nep_best.txt` keeps the parameters with the lowest (validation) loss, and `use_swa=True` averages the weights over the last epochs [[9]](#references).

The trained model is written as GPUMD's `nep.txt`, so it runs unchanged in GPUMD, NEP_CPU and LAMMPS.

## References

1. Z. Fan, Z. Zeng, C. Zhang, Y. Wang, K. Song, H. Dong, Y. Chen and T. Ala-Nissila, *Neuroevolution machine learning potentials: Combining high accuracy and low cost in atomistic simulations and application to heat transport*, Phys. Rev. B **104**, 104309 (2021). [doi:10.1103/PhysRevB.104.104309](https://doi.org/10.1103/PhysRevB.104.104309) — NEP1: the site-energy network, the forces and SNES training.
2. Z. Fan, *Improving the accuracy of the neuroevolution machine learning potentials for multi-component systems*, J. Phys.: Condens. Matter **34**, 125902 (2022). [doi:10.1088/1361-648X/ac462b](https://doi.org/10.1088/1361-648X/ac462b) — NEP2: element-pair dependent descriptor coefficients.
3. Z. Fan, Y. Wang, P. Ying, K. Song, J. Wang, Y. Wang, Z. Zeng, K. Xu, E. Lindgren, J. M. Rahm, A. J. Gabourie, J. Liu, H. Dong, J. Wu, Y. Chen, Z. Zhong, J. Sun, P. Erhart, Y. Su and T. Ala-Nissila, *GPUMD: A package for constructing accurate machine-learned potentials and performing highly efficient atomistic simulations*, J. Chem. Phys. **157**, 114801 (2022). [doi:10.1063/5.0106617](https://doi.org/10.1063/5.0106617) — NEP3: the Chebyshev radial functions and the 3- to 5-body angular components.
4. K. Song, R. Zhao, J. Liu, Y. Wang, E. Lindgren, Y. Wang, S. Chen, K. Xu, T. Liang, P. Ying, N. Xu, Z. Zhao, J. Shi, J. Wang, S. Lyu, Z. Zeng, S. Liang, H. Dong, L. Sun, Y. Chen, Z. Zhang, W. Guo, P. Qian, J. Sun, P. Erhart, T. Ala-Nissila, Y. Su and Z. Fan, *General-purpose machine-learned potential for 16 elemental metals and their alloys*, Nat. Commun. **15**, 10208 (2024). [doi:10.1038/s41467-024-54554-x](https://doi.org/10.1038/s41467-024-54554-x) — NEP4: one network per element.
5. J. F. Ziegler, J. P. Biersack and U. Littmark, *The Stopping and Range of Ions in Solids*, vol. 1, Pergamon, New York (1985) — the universal ZBL repulsion.
6. J. Liu, J. Byggmästar, Z. Fan, P. Qian and Y. Su, *Large-scale machine-learning molecular dynamics simulation of primary radiation damage in tungsten*, Phys. Rev. B **108**, 054312 (2023). [doi:10.1103/PhysRevB.108.054312](https://doi.org/10.1103/PhysRevB.108.054312) — ZBL in NEP.
7. I. Loshchilov and F. Hutter, *Decoupled weight decay regularization*, International Conference on Learning Representations (2019). [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) — AdamW.
8. S. J. Reddi, S. Kale and S. Kumar, *On the convergence of Adam and beyond*, International Conference on Learning Representations (2018). [arXiv:1904.09237](https://arxiv.org/abs/1904.09237) — AMSGrad.
9. P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov and A. G. Wilson, *Averaging weights leads to wider optima and better generalization*, Conference on Uncertainty in Artificial Intelligence (2018). [arXiv:1803.05407](https://arxiv.org/abs/1803.05407) — stochastic weight averaging.
