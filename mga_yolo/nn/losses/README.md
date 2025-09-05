# Loss

Kendall et al. (CVPR 2018) learn task weights via homoscedastic uncertainty: each task loss Li is scaled as exp(−si)·Li + si, where si=log σi² is a learnable parameter. This removes manual λ-tuning and typically stabilizes and improves joint training; weights adapt during training and outperform static grids.  
Unified Focal Loss (UFL) reduces Dice+CE’s parameter sprawl and handles input/output imbalance with three knobs {λ,δ,γ}: LsUF=λ·LmF+(1−λ)·LmFT, with the modified focal CE and focal-Tversky terms below. It improves recall-precision balance on imbalanced medical masks. 

Previous code: The trainer simply sums detection and segmentation losses with a single fixed λ (seg\_loss\_lambda), so gradients can dominate intermittently. 

Kendall weighting removes the single fixed λ and replaces it with learned task weights that adapt over training. This directly addresses the unstable trade-off you observed between detection and mask guidance.
UFL provides a stronger, imbalance-aware segmentation objective than BCE+Dice, with three interpretable hyperparameters and the Dice-like component derived from the modified Tversky index. It reduces rare-region under-learning while avoiding exploding Dice gradients. 

## Formalization

$$
\boxed{\;\mathcal{L}_{\text{total}}
= e^{-s_{\text{det}}}\,\mathcal{L}_{\text{det}} + s_{\text{det}}
  + e^{-s_{\text{seg}}}\,\mathcal{L}_{\text{seg}} + s_{\text{seg}}\;}
$$

with homoscedastic‐uncertainty logs $s_{\text{det}},s_{\text{seg}}\in\mathbb{R}$ learned with the model.

### Detection branch

Let $\mathcal{L}_{\text{box}},\mathcal{L}_{\text{cls}},\mathcal{L}_{\text{dfl}}$ be the YOLOv8 losses (IoU/box, BCE logits for classes, DFL). With scalar gains $\lambda_{\text{box}},\lambda_{\text{cls}},\lambda_{\text{dfl}}$ from `hyp`:

$$
\boxed{\;\mathcal{L}_{\text{det}}
= \lambda_{\text{box}}\,\mathcal{L}_{\text{box}}
+ \lambda_{\text{cls}}\,\mathcal{L}_{\text{cls}}
+ \lambda_{\text{dfl}}\,\mathcal{L}_{\text{dfl}}\;}
$$

### Segmentation branch: two options

Let scales $k\in\{P3,P4,P5\}$ with weights $a_k>0$, logits $z_k$, probabilities $p_k=\sigma(z_k)$, masks $y_k\in\{0,1\}^{B\times1\times H\times W}$, and a global multiplier $\lambda_{\text{seg}}>0$.

#### Option A — BCE + Dice (current)

$$
\mathcal{L}_{\text{BCE}}^{(k)} = \frac{1}{N}\sum_{i=1}^N
\operatorname{BCEWithLogits}\!\big(z_{k,i},y_{k,i}\big),
$$

$$
\mathcal{L}_{\text{Dice}}^{(k)} = 1-\frac{2\langle p_k,y_k\rangle + \epsilon}
{\|p_k\|_1 + \|y_k\|_1 + \epsilon},
$$

with mixture weights $w_{\text{bce}},w_{\text{dice}}>0$.

$$
\boxed{\;\mathcal{L}_{\text{seg}}
= \lambda_{\text{seg}}\sum_{k} a_k
\big( w_{\text{bce}}\,\mathcal{L}_{\text{BCE}}^{(k)}
    + w_{\text{dice}}\,\mathcal{L}_{\text{Dice}}^{(k)} \big)\;}
$$

#### Option B — Unified Focal Loss (UFL)

Define $p_{t,i}=
\begin{cases}
p_{k,i}, & y_{k,i}=1\\
1-p_{k,i}, & y_{k,i}=0
\end{cases}$,
class balance $\delta\in(0,1)$,
focus $\gamma\ge 0$,
and weight $w(y)=\delta$ if $y=1$ else $1-\delta$.

Modified focal CE (per scale):

$$
\mathcal{L}_{\mathrm{mF}}^{(k)}
=\frac{1}{N}\sum_{i=1}^N
\big(1-p_{t,i}\big)^{\,1-\gamma}\,
w(y_{k,i})\,
\operatorname{BCEWithLogits}\!\big(z_{k,i},y_{k,i}\big).
$$

Modified focal Tversky (per scale), with smoothing $\epsilon>0$:

$$
\mathrm{TP}_k=\langle p_k,y_k\rangle,\quad
\mathrm{FP}_k=\langle p_k,1-y_k\rangle,\quad
\mathrm{FN}_k=\langle 1-p_k,y_k\rangle,
$$

$$
\mathrm{mTI}_k=\frac{\mathrm{TP}_k+\epsilon}
{\mathrm{TP}_k+\delta\,\mathrm{FN}_k+(1-\delta)\,\mathrm{FP}_k+\epsilon},\qquad
\mathcal{L}_{\mathrm{mFT}}^{(k)}=\big(1-\mathrm{mTI}_k\big)^{\gamma}.
$$

Blend the two with $\lambda_{\mathrm{ufl}}\in[0,1]$:

$$
\boxed{\;\mathcal{L}_{\text{seg}}
= \lambda_{\text{seg}}\sum_{k} a_k\,
\Big(\lambda_{\mathrm{ufl}}\,\mathcal{L}_{\mathrm{mF}}^{(k)}
+(1-\lambda_{\mathrm{ufl}})\,\mathcal{L}_{\mathrm{mFT}}^{(k)}\Big)\;}
$$

Notes.
-  $\epsilon>0$ is the Dice/Tversky smoothing constant.
-  All sums/means are over batch and pixels of scale $k$.
-  The uncertainty terms $s_{\text{det}},s_{\text{seg}}$ are free scalars learned jointly; the effective task weights are $e^{-s_{\text{det}}}$ and $e^{-s_{\text{seg}}}$.


## Sources:

[1] Kendall, Gal, Cipolla. “Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.” CVPR 2018. Key concepts and formulation of exp(−s)·L + s. 

[2] Yeung et al. “Unified Focal loss: Generalising Dice and cross entropy-based losses to handle class imbalanced medical image segmentation.” CMIG 2022. Definitions of LmF, LmFT, and LsUF with {λ,δ,γ}. 


