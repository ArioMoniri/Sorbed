# Models, Datasets & the Data-Poverty Reality

This document is an honest accounting of the machine-learning landscape Sorbed operates in. Wound imaging is a **data-poor** field, and pressure-injury staging is the poorest corner of it. It is important to say this plainly at the outset: during the survey behind Sorbed's model choices, **no large, public, permissively-licensed pressure-injury *staging* dataset could be verified to exist.** The reputable staging studies in the literature are trained on private, single-center data. That reality shapes every recommendation below, and it is the reason Sorbed treats staging as decision support with explicit uncertainty rather than as a settled classification problem.

A disclaimer that applies to every number in this document: the benchmark figures cited here are **author-reported and, in most cases, single-center.** They have not been independently reproduced by the Sorbed project. Treat them as indicative of relative capability, not as guarantees of performance on your data.

## Two very different tasks

It helps to separate two problems that are often conflated:

1. **Wound boundary segmentation** — drawing the outline of the wound. This task is comparatively **mature**. Public data is reasonably good, and models routinely reach Dice scores in the **0.85–0.93** range.
2. **Tissue-type semantic segmentation** — labeling the bed as granulation, slough, eschar, epithelial tissue, and so on. This is **hard**. Public data is tiny, class boundaries are genuinely ambiguous, and the classic failure mode is **slough being confused with granulation** (yellow-tinged granulation and pink-tinged slough sit close together in color space).

Sorbed's architecture reflects this split: it is confident about *where* the wound is and appropriately humble about *what every pixel of it is made of*.

## Segmentation models

| Model | License | Notes |
|---|---|---|
| [SAM 2.1 Hiera](https://github.com/facebookresearch/sam2) | Apache-2.0 | Promptable and class-agnostic; excellent for interactive use and for bootstrapping labels. Does not know what a wound is on its own. |
| [MedSAM / MedSAM2](https://github.com/bowang-lab/MedSAM) | Apache-2.0 | Oriented toward CT/MR and 3D volumes — **not ideal for 2D RGB clinical photos**. |
| [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) | Apache-2.0 | Self-configuring; trains from scratch. Highest automatic ceiling of anything here, but needs a GPU and a real labeled dataset. |
| DeepLabV3+ (torchvision) | BSD | Solid, well-supported semantic segmentation backbone. |
| [segmentation_models.pytorch (SMP)](https://github.com/qubvel-org/segmentation_models.pytorch) | MIT | U-Net / DeepLab families with ImageNet encoders. A MobileNet encoder runs in **real time on CPU**. |
| WSNet | (wound-specific) | Reports Dice 0.847. |
| FUSegNet | (wound-specific) | Reports 92.7% Dice on foot ulcers. |
| DFUTissueSegNet | (wound-specific) | Tissue segmentation, trained on very little data. |

One further data point worth recording: the **WoundAmbit** benchmark found that **SegNeXt** and **SegFormer** outperformed a plain U-Net, and that **all evaluated models ran at ≥1 image/second on CPU.** That last finding matters for deployment — it means CPU-only serving is genuinely viable for this workload, not just a fallback.

## Datasets

The table below is deliberately candid about size and license, because those two facts constrain what can actually be built and shipped.

| Dataset | Size | Content | License / availability |
|---|---|---|---|
| AZH Chronic Wound | 1,109 images | Binary wound masks | Public |
| FUSeg 2021 | 1,210 images | Wound masks | Public |
| [DFUC 2022](https://dfuc2022.grand-challenge.org/) | 4,000 images | Diabetic foot ulcer | **Non-commercial** license agreement |
| DFUTissue | 110 images | Tissue labels | Small |
| 6-tissue set (arXiv:2502.10652) | 147 images | Six tissue classes | Small |
| WoundSeg / WSNet | 2,686 images | 8 wound types, incl. pressure ulcer | — |
| Medetec | — | Clinical stock photos | **No formal license** ("free stock") |

The gap is glaring and worth naming again: **there is no verifiable public, permissively-licensed pressure-injury *staging* dataset.** This is the single biggest obstacle in the field. Everything that exists publicly is either boundary segmentation, non-pressure wounds (foot ulcers), or tiny tissue sets — and the published staging work relies on private single-center collections that cannot be redistributed.

## Staging models — an honest reading

The reported accuracy figures for staging classifiers look impressive until you read the fine print:

- **DenseNet121 ~93.7%** and **ResNet18 ~92.4%** on an **853-image single-center** set (JMIR, 2025). These numbers are **inflated by augmentation** and reflect one institution's imaging.
- **YOLOv8m** for mobile deployment: **84.6%**.
- **Thermal imaging: 95.45%** — but this requires a thermal camera, which most deployments will not have.

The context that deflates all of these: **human raters agree with one another only 23–58% of the time** on staging. A model that "agrees with the labels" is agreeing with a noisy ground truth. More fundamentally, **staging from a single 2D image is intrinsically hard.** Depth is ambiguous from RGB, and the **Unstageable and DTPI categories are, in principle, unresolvable from surface RGB alone** — Unstageable means the depth is hidden by definition, and DTPI's severity lives beneath intact skin.

The conclusion Sorbed draws is not "staging is impossible" but "staging must ship as **decision support with uncertainty**." Low-confidence results, and specifically **Unstageable and DTPI, are routed to a clinician** rather than reported as confident predictions.

## Tissue classification approaches

There are two broad families, with a real trade-off between them:

- **Color-space methods** (HSV or Lab thresholds, the Red-Yellow-Black proxy) are **cheap, interpretable, and brittle.** They run anywhere, need no training data, and can be explained pixel by pixel — but they break under variable lighting, skin tone, and camera characteristics.
- **Learned semantic segmentation** is **state of the art but data-limited.** Where it has enough data it wins clearly — Swift's SmartTissue reports **94%**, but it is **proprietary** and not available to build on.

Given the data poverty, Sorbed reports tissue as **percentage-of-area estimates with confidence, not hard per-pixel claims.** Saying "roughly 60% granulation, 30% slough, 10% eschar, moderate confidence" is both more honest and more clinically useful than asserting a crisp boundary the data cannot support.

## Recommended stack

Putting the above together, the recommended Sorbed model stack is:

- **Automatic boundary segmentation:** [SMP](https://github.com/qubvel-org/segmentation_models.pytorch) DeepLabV3+ or U-Net with an EfficientNet or MobileNet encoder, fine-tuned on **FUSeg + AZH**.
- **Interactive segmentation:** [SAM 2.1](https://github.com/facebookresearch/sam2) for clinician-in-the-loop prompting and label bootstrapping.
- **Serving:** PyTorch → ONNX → ONNX Runtime behind FastAPI, **CPU-first with oneDNN**, scaling out to Triton/GPU when available.
- **Medical metrics and transforms:** [MONAI](https://github.com/Project-MONAI/MONAI) (Apache-2.0).
- **Explainability:** [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) (MIT) and Captum (BSD) for the vision models, plus SHAP for the feature-based staging head.

Underneath all of this sits a **weight-free classical backend** that guarantees end-to-end operation with no model downloads at all — the classical path (from the [uwm-bigdata wound-segmentation](https://github.com/uwm-bigdata/wound-segmentation) lineage of approaches and simple color-space tissue analysis) always produces a result, even fully offline.

## Licensing bottom line

The **code dependencies are uniformly permissive** — Apache-2.0, MIT, and BSD throughout. The constraints that actually matter are on **data**, not software: **DFUC is non-commercial**, and **Medetec carries no formal license.** Before any commercial deployment, **verify each dataset's license individually.** A permissive code stack does not make a non-commercial dataset commercially usable.

## Provenance & model cards

Sorbed **never vendors model weights.** Each model is:

- downloaded only on **explicit request**,
- **sha256-verified** on download,
- and **recorded, with its license, in a model registry** that functions as a lightweight model card for every artifact the system can load.

Because a **weight-free classical backend is always available**, the system remains fully functional even when no weights have been fetched — provenance discipline never comes at the cost of a working pipeline.

---

## Running with real HuggingFace models (SAM / MedSAM)

Sorbed ships a real learned segmentation backend that refines the wound mask with
a promptable Segment-Anything model from the HuggingFace Hub. The weight-free
classical proposal supplies a bounding-box prompt; the model returns a precise
mask and its own predicted IoU, which Sorbed uses as the segmentation confidence
(a genuine model-derived value).

```bash
pip install -e '.[hf]'                        # torch + transformers + huggingface-hub
export SORBED_SEGMENTATION_BACKEND=hf_sam
export SORBED_HF_MODEL_ID=facebook/sam-vit-base          # or a medical variant:
# export SORBED_HF_MODEL_ID=flaviagiammarino/medsam-vit-base
sorbed analyze wound.jpg --mm-per-px 0.15 --out reports
```

Verified real, permissively-licensed Hub models usable as drop-in prompts:

| Model id | What it is | License |
|---|---|---|
| `facebook/sam-vit-base` | Segment Anything (ViT-B), promptable, class-agnostic | Apache-2.0 |
| `flaviagiammarino/medsam-vit-base` | MedSAM, box-prompted medical SAM | Apache-2.0 |

Weights download on first use and are cached under the HuggingFace cache; nothing
is vendored, and the resolved Hub commit is recorded in the report's provenance.

> **Network note.** The backend needs outbound access to `huggingface.co`. Some
> managed/sandboxed environments firewall it; in that case pre-download the model
> in a networked environment (`huggingface-cli download <id>`) and point
> `HF_HOME` at the shared cache, or run Sorbed where the Hub is reachable. Sorbed
> never fabricates a result when a model cannot be loaded — it raises a clear
> error and the classical backend remains available offline.
