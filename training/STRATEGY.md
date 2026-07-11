# Sorbed model & learning strategy — decision of record

This document is the **decision of record** for how Sorbed's learned backends are
trained, kept honest, and improved over time. It is deliberately opinionated: the
pipeline below was chosen, and the trainers/modules in `training/` and
`src/sorbed/feedback/` are built to exactly this shape. Read it before touching a
trainer, and read [`training/RUNBOOK.md`](RUNBOOK.md) for the operational how-to.

The shipped CPU pipeline never depends on any of this — the classical backend
always works offline. Everything here concerns the *optional* learned backends
and the roadmap that upgrades them.

---

## 1. The two deployed models

Only two learned models are actually served, both exported to **ONNX for CPU
inference**:

| Model | Role | Trainer | Config |
| --- | --- | --- | --- |
| **SegFormer MiT-b3** | wound-area segmenter | `training/train_seg.py` | `training/configs/seg_segformer.yaml` |
| **ConvNeXt-V2** | pressure-injury stage grader | `training/train_grade.py` | `training/configs/grade_convnextv2.yaml` |

The segmenter is trained on real public foot-ulcer benchmarks (AZH / MICCAI-2021
FUSeg); the U-Net++/EfficientNet recipe in
`training/configs/seg_unetpp_effnet.yaml` is the default server bring-up config
because it fits one ~40 GB MIG slice comfortably and is the fastest path to a
verified ONNX export. SegFormer MiT-b3 is the target deployed segmenter.

Everything else in this document exists because **grading has no data**.

---

## 2. Why teacher–student for grading (no open graded corpus)

There is no large, public, permissively-licensed pressure-injury *staging*
dataset — this was checked, not assumed (see [`docs/MODELS.md`](../docs/MODELS.md)).
Worse, staging ground truth is intrinsically noisy: human raters agree only
~23–58% of the time. You cannot supervised-train a grader on data that does not
exist.

So grading is bootstrapped by **teacher–student distillation**
(`training/teacher_student.py`, config `training/configs/teacher_student.yaml`):

- The **`DirectiveTeacher`** encodes NPIAP/EPUAP staging doctrine and the
  HD_T86 directive pack (`scripts/build_directive_pack.py`) as a deterministic
  rule engine. Given the morphometric/tissue features Sorbed already computes, it
  emits a **soft label** over stages — and critically it **abstains** when the
  evidence is ambiguous (e.g. a depth-dependent Stage 3-vs-4 call an image simply
  cannot resolve). Abstention is a feature: it refuses to manufacture a label
  where a clinician would also hedge.
- The **ConvNeXt-V2 student** learns those soft labels (distillation loss over
  the non-abstained rows). It generalises the rules into a visual model instead
  of memorising a fabricated hard-label set.

This is core, not a stopgap: with no graded corpus, a rule-teacher that abstains
is the most honest supervisory signal available, and distillation is how that
signal becomes a fast image model.

---

## 3. Why MedSAM is a mask factory, not the deployed segmenter

Several staged-but-**maskless** collections exist (PIID stage labels,
DFUC-2021, Medetec aetiology images). They carry the labels grading needs but no
segmentation masks, so they cannot be used for joint seg+grading as-is.

`training/finetune_medsam.py` fine-tunes MedSAM and runs it as an **offline mask
factory**: it auto-generates wound masks for those sets so they become usable
training data. MedSAM is **not** deployed — it is a promptable, heavier model
whose only job here is to manufacture masks once, offline. The deployed segmenter
remains the ONNX SegFormer of §1. Keeping these roles separate is deliberate: the
mask factory can be as heavy as it likes because it never runs at inference time,
and the deployed segmenter stays small and CPU-friendly.

---

## 4. Human-in-the-loop = the second teacher

The rule teacher (§2) is teacher #1. Real clinicians are **teacher #2**, and
their signal outranks the rules.

At runtime the feedback loop lives in `src/sorbed/feedback/`
(`records.py`, `store.py`, `emit.py`) and the `sorbed feedback` CLI group
(`src/sorbed/cli/feedback_cmd.py`):

- Every analysis emits an inference record. A nurse can submit a **grade**, and a
  reviewer can submit an **HQ correction** (mask fix + confirmed stage). These
  land in the feedback store as ground truth for *exactly the images the system
  got in front of a clinician*.
- `sorbed feedback export` JOINs the feedback log onto the inference log and
  writes a training manifest CSV. Human HQ/physician corrections take
  **precedence** over the rule-teacher's own label for the same row (the role
  precedence lives in `feedback_cmd.py`).

Because these corrections are weighted above rule-teacher labels, the student
progressively learns from real clinical judgment on real deployed images — the
one signal a rule engine can never provide.

---

## 5. Continual learning: how the loop closes end-to-end

`training/continual.py` (config `training/configs/continual.yaml`) consumes the
exported manifest and performs a **replay-buffered continual fine-tune** of the
currently-promoted grader:

1. **Harvest.** Read the new joined rows from `sorbed feedback export`.
2. **Replay.** Mix them with a stratified sample of previously-confirmed rows
   from the replay manifest, capped per class, so a small weekly batch cannot
   overfit the latest round or forget the distilled base (elastic anchoring to
   the base weights + a small LR).
3. **Fine-tune** the ConvNeXt student on the combined set, weighting human
   corrections above rule-teacher labels.
4. **Gate.** `promotion_decision` refuses to promote unless the candidate shows
   **no regression** on a frozen hold-out in balanced accuracy, quadratic-weighted
   kappa (QWK), or calibration (ECE). Each check is one-sided; ties go to the base.
5. **Promote.** On a pass it writes `promoted.pt`, **exports ONNX**
   (`training/models.export_onnx`) with a recorded sha256, and **appends** the
   round's confirmed rows to the replay manifest so they anchor future rounds.

If the gate fails, nothing ships — the previously-promoted ONNX stays live. This
is the safety property that lets an unattended weekly loop be trusted.

---

## 6. Federated-learning-ready (designed now, wired later)

`training/fl_client.py` packages each promoted continual round as a **client
delta** (`ClientUpdate`, `compute_delta` = `fine_tuned − base`) so a future
FedAvg/FedProx server can average updates **across hospitals without patient data
ever leaving the site** — only the parameter delta and non-PHI aggregate stats
travel. `apply_delta` and `fedavg_aggregate` implement the inverse and the server
step. FedProx is preferred over plain FedAvg because hospital data is heavily
non-IID (different stage mix, camera, skin-tone distribution); the packaging is
identical either way. This seam is **built now and exercised in tests, but not
wired to a live server** — it is the roadmap capability, designed so wiring it
later requires no change to the trainers.

---

## 7. End-to-end flow

```
                         ┌──────────────────────────────────────────────────┐
                         │  OFFLINE BOOTSTRAP (one-time / periodic)          │
                         │                                                  │
  NPIAP/EPUAP doctrine ──▶ DirectiveTeacher (teacher_student.py)            │
  + HD_T86 pack           │   soft labels, ABSTAINS when ambiguous          │
                         │            │ distill                             │
  staged-but-maskless ──▶ MedSAM mask factory (finetune_medsam.py)         │
  sets (PIID/DFUC/…)      │   auto-masks ─────────┐                         │
                         │            ▼           ▼                         │
                         │      ConvNeXt-V2 student  ◀── train_grade.py     │
                         │      SegFormer segmenter  ◀── train_seg.py       │
                         └───────────────┬──────────────────────────────────┘
                                         │ export ONNX
                                         ▼
   ┌─────────────────────── RUNTIME (CPU, offline) ───────────────────────┐
   │  capture ─▶ analyze ─▶ emit record + mask + stats                    │
   │                              (src/sorbed/feedback: records/store/emit)│
   │                                   │                                   │
   │        nurse grade / HQ correction ▼  (sorbed feedback record|submit) │
   │                          feedback store (ground truth)                │
   └───────────────────────────────────┬───────────────────────────────────┘
                                        │ sorbed feedback export (JOIN)
                                        ▼
                             training manifest CSV
                                        │  + replay manifest (past rounds)
                                        ▼
        replay-buffered continual fine-tune  (training/continual.py)
                                        │
                             promotion gate (no regression:
                             balanced-acc / QWK / ECE on frozen holdout)
                              ┌─────────┴─────────┐
                         FAIL │                   │ PASS
                       keep live ONNX      promote ▶ export ONNX (+sha256)
                                                   │  append rows to replay
                                                   ▼
                                    (future) FL client delta
                                    training/fl_client.py ─▶ FedAvg/FedProx server
```

---

## 8. Honest limitations

- **Grading accuracy is bounded by the rule teacher and the clinicians who
  correct it.** Distillation cannot exceed the quality of its supervision, and
  staging ground truth is inherently noisy (§2). Sorbed therefore ships grading as
  decision support with explicit uncertainty, never as a diagnosis.
- **Foot ulcers ≠ pressure injuries.** The public segmentation benchmarks are
  foot ulcers; site, tissue and depth differ. The honest requirement is training
  on an in-house pressure-injury set, not a foot-ulcer fine-tune. The FUSeg path
  is the reproducible *bring-up*, not the clinical endpoint.
- **MedSAM masks are machine-generated.** Auto-masks carry MedSAM's biases and
  should be spot-checked before they anchor a grading round.
- **The continual gate protects against regression, not against distribution
  shift you never measured.** The hold-out must be curated and refreshed by hand;
  an unrepresentative hold-out gives false confidence.
- **Federated learning is designed, not deployed.** `fl_client.py` produces and
  aggregates deltas in tests; there is no live cross-hospital server, and secure
  aggregation / differential privacy are noted as wrappers, not yet enforced.
