# Modules

## Segmentation Head

Forward pass only. It maps one FPN feature map (e.g., P3/P4/P5) to a same-size mask logit map with a shallow Conv–Norm–Activation–Conv stack. No upsampling, no post-processing, no special inference path.

Process, shapes, and math
Input $x\in\mathbb{R}^{B\times C_{\text{in}}\times H\times W}$.

1. $1\times1$ projection (channel mixing, shape-preserving):

$$
z_1 = \text{Conv}_{1\times1}(x;\;W_1)\,,\quad
W_1\in\mathbb{R}^{C_h\times C_{\text{in}}\times1\times1}
$$

Output $\in\mathbb{R}^{B\times C_h\times H\times W}$.

2. Normalization (optional):

* BatchNorm2d: $\hat z = \gamma\frac{z_1-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}+\beta$
* Channel-last LayerNorm: permute to $NHWC$, apply LN over $C_h$, permute back.

3. Activation (default SiLU): $a = z \cdot \sigma(z)$.

4. Spatial dropout (optional): $a'=\text{Dropout2d}(a)$.

5. $3\times3$ head to logits:

$$
y = \text{Conv}_{3\times3}(a';\;W_2)\,,\quad
W_2\in\mathbb{R}^{C_{\text{out}}\times C_h\times3\times3}
$$

Output $y\in\mathbb{R}^{B\times C_{\text{out}}\times H\times W}$.

```
P3: x ∈ B×C_in×H×W
  → [Branch A] 1×1 (C_in→C_h) → GN → SiLU → SE(C_h) → DW-3×3 (d=1) → PW-1×1 (C_h→C_out=1)
  → [Branch B] 1×1 (C_in→C_h) → GN → SiLU → DW-3×3 (d=2) → PW-1×1 (C_h→1)
  → Fuse: logits = A + B
  → Bias init to logit prior p0
Output: B×1×H×W (logits)
Skip (optional): 1×1 (C_in→1) added into logits for residual guidance.
```
