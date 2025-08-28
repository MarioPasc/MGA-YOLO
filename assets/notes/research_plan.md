# 1) Research plan (paper/congress-ready)

## Hypothesis & scope

Mask-guided attention (external attention) applied at P3/P4/P5 improves detection by (i) sharpening geometry at high-resolution levels and (ii) reinforcing semantics at coarse levels. We will **not** share weights across scales (scale-specific specialization) and keep Detect’s API unchanged (same shapes). YOLOv8 (same size as our model: n/s/m) is the main baseline. ([Ultralytics Docs][1], [CVF Open Access][2])

## Datasets (multi-domain, detection + masks when available)

* **Coronary angiography**: ARCADE-style/stenosis & vessel masks (Popov et al., 1,500 frames with vessel segments + stenosis locations). ([Nature][3])
* **Retina**: DRIVE, CHASE-DB1, STARE (vessel masks; use boxes for lesions/tools when available or synthesize boxes from masks for ablations). Large cross-dataset analyses exist. ([Nature][4])
* **Colonoscopic polyps**: Kvasir-SEG, CVC-ClinicDB, ETIS; plus **EDD2020** (joint detection + segmentation). ([web-backend.simula.no][5], [PMC][6], [Dataset Ninja][7])
  (If a dataset lacks boxes, we still use its masks to train the mask head and report detection on the ones that include boxes; segmentation metrics are always reported.)

## Model variants (ablations; all drop-in, shape-preserving)

1. **Baselines**: YOLOv8; +DMM; +HSMA (from your prior write-up).
2. **New blocks** (each at P3/P4/P5; no weight sharing):

   * **Masked ECA (channel)**, **FiLM/SPADE-Norm (mask-conditioned)**, **Partial-Conv gate**, **Boundary-aware gate**, **MG-DCNv2 (P3→P4)**, **Masked cross-attention (lite)**. ([CVF Open Access][8], [arXiv][9])
3. **Scale placement study**: P3-only / P4-only / P5-only / P3+P4 / P4+P5 / P3+P5 / all.
4. **Fusion policy**: with / without boundary branch; with / without stop-grad on mask; soft vs binarized masks.
5. **Complexity control**: parameter-matched versions (add 1×1 convs to baselines to equalize params/FLOPs) and latency profiling.

## Training & evaluation

* **Loss**: Standard YOLOv8 detection losses + λ·(Dice or Focal) for masks; tune λ. ([Ultralytics Docs][1])
* **Metrics (detection)**: mAP@\[.5:.95], AP$_S$/AP$_M$/AP$_L$; **(segmentation)**: Dice/IoU **and** **Boundary-IoU / BF-score** to capture edge fidelity (important for vessels/polyps). ([arXiv][10], [MathWorks][11])
* **Compute**: Params, FLOPs, and measured latency on a fixed GPU batch=1/8.
* **Generalization**: train on one dataset, test zero-shot on another (e.g., train Kvasir-SEG → test ClinicDB/ETIS; train DRIVE → test CHASE/STARE). ([PMC][12])
* **Robustness**: occlusion, blur, noise, small-object stress (crop-resize), and sparse-mask stress (random erode of M).
* **Cross-scale evidence** (see §2): CKA/linear probes/grad-CAM/transform-equivariance to ground claims about “geometry @P3 vs semantics @P5”. ([Proceedings of Machine Learning Research][13], [arXiv][14])

## Reporting (paper-style figures & tables)

* **Main table**: mAP / Dice / Boundary-IoU vs Params/FLOPs/Latency across variants.
* **Ablation figure**: bars by scale placement; **Pareto plot** (mAP vs latency).
* **Qualitative**: side-by-side detections + mask overlays; **boundary zoom-ins**.
* **Mechanistic**: CKA heatmaps across layers; linear-probe accuracy curves; Grad-CAM panels (see §2). ([Proceedings of Machine Learning Research][13], [arXiv][14], [CVF Open Access][15])

---

# 2) How to *prove* “P3 = geometry-aware; P5 = semantic-heavy” (scientific & figure-ready)

We cannot merely assert it; we can **measure** it with four complementary tests:

### (A) Linear probes (predictors trained *on frozen features*)

Train tiny linear heads from each scale’s features to:

* **Edge/boundary maps** (from GT masks’ contours) → “geometry score”.
* **Class labels** (box-level or image-level) → “semantic score”.
  Expectation: P3 yields higher boundary-prediction accuracy; P5 yields higher class accuracy. Plot accuracy vs scale. ([arXiv][14])

### (B) Transform sensitivity (equivariance/invariance analysis)

Apply controlled **small translations/rotations/scales** to inputs; measure **feature-space CKA similarity** pre/post transform for each scale. Geometry-aware features should **change predictably** (equivariant; lower CKA under shifts); semantic features should be **more invariant** (higher CKA). Provide layer-wise CKA curves per transform. ([PMC][16], [Proceedings of Machine Learning Research][13])

### (C) Concept/region attribution

Use **Grad-CAM** w\.r.t. the Detect head **but read out activations at P3 vs P5** (tap before Detect). Quantify the **Boundary Activation Ratio** = mean activation in a trimap rim / mean activation inside the mask. Expect higher ratios at P3. Aggregate across datasets; show violin plots. ([CVF Open Access][15])

### (D) Boundary-quality correlation

Compute **Boundary-IoU** / **BF-score** *per image* using the predicted masks, and correlate them with **AP gains** contributed by P3 vs P5 interventions (ablate P3-only vs P5-only). If P3 gates improve boundary metrics more and co-vary with AP on small objects, that supports “geometry\@P3”. Scatter with Pearson r. ([arXiv][10], [MathWorks][11])

**Figure for the paper (one page):**

1. **Left**: CKA-vs-transform curves for P3/P4/P5 (translation/rotation).
2. **Middle**: Linear-probe bars (edge vs class) per scale.
3. **Right-top**: Grad-CAM heatmaps at P3/P5 on the same image.
4. **Right-bottom**: Boundary-metric vs AP$_S$ scatter across variants.
   (Background: FPN aims to propagate semantics to all scales, but scales still differ in content/resolution; BiFPN explicitly learns unequal scale weights—so testing specialization is warranted.) ([CVF Open Access][2])

---

# 3) Final summary table: modules & scale-wise recommendations

| Module                            | Core operation                                    | ΔParams / ΔFLOPs      | Robustness to mask noise | Where it shines                         | Recommended scales                            |
| --------------------------------- | ------------------------------------------------- | --------------------- | ------------------------ | --------------------------------------- | --------------------------------------------- |
| **Masked ECA**                    | Channel weights from masked GAP; $F'=s\odot F$    | **Tiny / tiny**       | Medium                   | Cheap channel selectivity               | P4, P5 ([CVF Open Access][8])                 |
| **FiLM / SPADE-Norm**             | Mask-conditioned $\gamma,\beta$ on normalized $F$ | Small / small         | High                     | Injects spatial mask without hard zeros | P3 (detail), P4 ([CVF Open Access][17])       |
| **Partial-Conv gate**             | Conv on $F\odot M$ with renorm                    | Small-med / small-med | **High**                 | Robust at rims/holes                    | P3 (boundaries) ([CVF Open Access][18])       |
| **Boundary-aware gate**           | Conv on $[M,\text{edge}(M)]$ → boost rim          | **Tiny / tiny**       | Medium-High              | Tight boxes; improves Boundary-IoU      | P3→Detect ([arXiv][10])                       |
| **MG-DCNv2**                      | Deformable conv with mask-modulated offsets       | Med / **med-high**    | High                     | Geometry-adaptive sampling              | P3 (then P4 if gains) ([CVF Open Access][19]) |
| **Masked cross-attention (lite)** | Attend within $\{M>0\}$ only                      | Med / med             | Medium                   | Global context *inside* region          | P3 (optionally P4) ([CVF Open Access][20])    |
| **SimAM (masked stats)**          | Param-free energy attention (masked)              | 0 / tiny              | Medium                   | Free boost; easy to add                 | Any scale ([arXiv][21])                       |

**Design rule (critical view of your proposal):**
Use the **same interface** at P3/P4/P5 but **different parameterization** (no weight sharing). Emphasize **spatial/geometry** at P3 (FiLM/SPADE + Partial-Conv + optional MG-DCNv2), **channel & RF** at P5 (Masked ECA ± SK), and a **balanced** mix at P4. This respects the multi-scale specialization observed in pyramid detectors and supported by BiFPN’s learned scale weighting. ([CVF Open Access][2])

---

## Minimal execution roadmap (1–2 pages in your Methods)

1. **Phase-1 (cheap, solid base):** Masked ECA (P4/P5) + FiLM-Norm (P3).
2. **Phase-2 (geometry):** + Boundary-aware gate (P3) → + Partial-Conv (P3).
3. **Phase-3 (heavier):** + MG-DCNv2 (P3). If justified, add to P4.
4. **Phase-4 (context):** + Masked cross-attention (P3).
5. **Mechanistic study:** run §2(A–D) and include the composite figure.

This plan gives you a **computationally ascending** series of improvements with strong, testable evidence for your “different blocks per scale” choice, while keeping Detect’s inputs untouched.

[1]: https://docs.ultralytics.com/models/yolov8/ "Explore Ultralytics YOLOv8"
[2]: https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf "Feature Pyramid Networks for Object Detection"
[3]: https://www.nature.com/articles/s41597-023-02871-z "Dataset for Automatic Region-based Coronary Artery ..."
[4]: https://www.nature.com/articles/s41598-022-09675-y "State-of-the-art retinal vessel segmentation with ..."
[5]: https://web-backend.simula.no/sites/default/files/publications/files/mmm_2020_kvasir_seg_debesh.pdf "Kvasir-SEG: A Segmented Polyp Dataset"
[6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11458519/ "A complete benchmark for polyp detection, segmentation ..."
[7]: https://datasetninja.com/edd2020 "EDD2020 Dataset"
[8]: https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf "ECA-Net: Efficient Channel Attention for Deep ..."
[9]: https://arxiv.org/abs/1804.07723 "Image Inpainting for Irregular Holes Using Partial ..."
[10]: https://arxiv.org/abs/2103.16562 "Boundary IoU: Improving Object-Centric Image Segmentation Evaluation"
[11]: https://www.mathworks.com/help/images/ref/bfscore.html "bfscore - Contour matching score for image segmentation"
[12]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9945716/ "Segmentation of polyps based on pyramid vision ..."
[13]: https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf "Similarity of Neural Network Representations Revisited"
[14]: https://arxiv.org/abs/1610.01644 "Understanding intermediate layers using linear classifier ..."
[15]: https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf "Grad-CAM: Visual Explanations From Deep Networks via ..."
[16]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6510825/ "Understanding Image Representations by Measuring Their ..."
[17]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_Semantic_Image_Synthesis_With_Spatially-Adaptive_Normalization_CVPR_2019_paper.pdf "Semantic Image Synthesis With Spatially-Adaptive ..."
[18]: https://openaccess.thecvf.com/content_ECCV_2018/papers/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.pdf "Image Inpainting for Irregular Holes Using Partial ..."
[19]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Deformable_ConvNets_V2_More_Deformable_Better_Results_CVPR_2019_paper.pdf "Deformable ConvNets V2: More Deformable, Better Results"
[20]: https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf "Masked-Attention Mask Transformer for Universal Image ..."
[21]: https://arxiv.org/abs/1612.03144 "[1612.03144] Feature Pyramid Networks for Object Detection"
