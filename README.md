<div align="center">

# Seeing the Sound of an Ear: <br>End-to-End Personalized HRTF Prediction from Sparse Pinna Photographs

**Rasul Alakbarli** · **Clément Marie**

*Huawei 2024 Munich Tech Arena — Media Technology Competition, finals in Munich, Germany*

🥈 **2nd prize** among 120 teams from 88 European universities

[**Overview**](#abstract) · [**Method**](#3-method) · [**Results**](#43-results) · [**Usage**](#repository) · [**References**](#references)

</div>

---

<p align="center">
  <img src="assets/fig1_teaser.svg" width="100%" alt="Teaser: from a handful of ear photos, one network predicts the full-sphere personalized HRTF."/>
</p>

<p align="center"><sub><b>Figure 1.</b> From a handful of grayscale ear photographs (19, 7, or even 3 views per ear), a single network directly regresses the listener's complete head-related impulse response set — 793 spatial directions × 2 ears — with no anechoic chamber, 3-D scan, or manual measurement in the loop.</sub></p>

## Abstract

> Head-Related Transfer Functions (HRTFs) encode how a listener's head, torso, and — above all — outer ears (pinnae) filter incoming sound as a function of direction, and are the key to convincing personalized spatial audio. Acoustically measuring an individual HRTF requires a treated room, a calibrated loudspeaker array, and tens of minutes of a motionless subject, which confines personalization to the laboratory. We present an end-to-end deep network that regresses a listener's **complete HRTF magnitude set — 793 directions × 2 ears — directly from a sparse, variable-size collection of grayscale pinna photographs**, bypassing the 3-D reconstruction, acoustic simulation, and anthropometric measurement stages of prior pipelines. A shared ResNet-50 encoder embeds each view; a bidirectional LSTM fuses the resulting *variable-length* sequence of view embeddings into a single identity code, so that one trained model serves all three challenge regimes (19, 7, or 3 images per ear) without retraining; and a second, *spatial-expansion* LSTM unrolls that code into per-output descriptors from which a shared MLP emits full-resolution head-related impulse responses in a single forward pass. The network is trained end-to-end on a differentiable implementation of its evaluation criterion — a critical-band-weighted Mean Spectral Distortion — so that every gradient step optimizes the perceptually weighted error it is judged on. The submitted model reached a Mean Spectral Distortion of **−63.6 dB**, earning **2nd prize** among 120 teams at Huawei's Munich Tech Arena.

## 1. Introduction

Human listeners localize sound with astonishing precision using just two ears. The physics behind this ability is captured by the Head-Related Transfer Function: the direction-dependent acoustic filter formed by the torso, head, and pinna on the path to each eardrum [1]. The pinna's folds imprint direction-specific spectral notches and peaks — the dominant cue for elevation and front/back disambiguation [2, 3] — and because every pinna is unique, every HRTF is too. Rendering spatial audio through *someone else's* HRTF is typically found to degrade elevation accuracy and front/back confusion rates relative to individualized rendering, and to reduce externalization, though the size of the gap varies with listener adaptation, head tracking, and reverberation [1, 5].

The bottleneck is acquisition. The gold standard measures each listener in an acoustically treated room with a loudspeaker arc and in-ear microphones — the SONICOM protocol, for instance, samples 793 directions per subject [23] — requiring specialized hardware, trained operators, and typically tens of minutes of subject time. The main measurement-free alternative reconstructs the listener's geometry (via 3-D scanning or photogrammetry) and numerically solves the wave equation with boundary-element methods [6, 7, 8]; this is accurate but requires watertight meshes and minutes-to-hours of compute [13]. Learning-based shortcuts have so far predicted HRTFs from hand-measured anthropometrics [4, 14], from single ear images *combined with* anthropometrics [10, 15], or from images via intermediate ear-shape models that still feed an acoustic simulator [9]. What would actually scale to consumer devices is simpler than all of these: **point a camera at each ear, take a few photos, get your HRTF.**

This work — built for Huawei's Munich Tech Arena HRTF challenge on the SONICOM dataset [23] — takes exactly that route. A single network consumes a *set* of grayscale pinna photographs of both ears and directly outputs the listener's complete head-related impulse-response tensor (793 directions × 2 ears × 256 taps) in SOFA format [28]. Four design decisions distinguish the system:

- **End-to-end photographic regression.** No 3-D reconstruction, no BEM simulation, no anthropometric tape measure, no acoustic capture. The mapping from pixels to full-sphere acoustics is learned as one differentiable function — pretrained visual features and gradient descent replace the entire classical pipeline (§3.2).
- **One model for any number of views.** The challenge evaluates three data-availability regimes — 19, 7, or 3 images per ear. Instead of training one model per regime, we treat the per-view CNN embeddings as a sequence and fuse them with a bidirectional LSTM [30, 31], whose summary state is independent of sequence length. The *same trained weights* handle all three tasks without retraining (§3.3). We report a single aggregate score and did not retain a per-regime breakdown, so we make no claim about how accuracy varies with view count.
- **Joint full-sphere decoding.** Rather than querying one direction at a time, as coordinate-conditioned neural-field approaches do [18, 19, 20], a recurrent *spatial-expansion* decoder unrolls the listener's 2048-dim identity code into all 1,586 output channels in a single pass, at roughly 1/40th the parameters a dense head would need (§3.4).
- **The evaluation metric is the loss.** We train directly on a differentiable Mean Spectral Distortion whose frequency weighting is inversely proportional to the auditory critical bandwidth [26, 27], and whose spatial support is the ±30° elevation band specified by the challenge protocol. The optimizer therefore spends its capacity exactly where the benchmark scores (§3.5).

Two further details matter in the low-data regime of 90 training subjects: augmentations are drawn *once per subject* and applied identically to every view, so the cross-view consistency the aggregator relies on is never destroyed (§3.6); and the ImageNet-pretrained encoder is surgically adapted to single-channel input rather than retrained from scratch (§3.2).

## 2. Related Work

**Measured and simulated HRTFs.** Direct measurement remains the reference standard and underlies all major HRTF corpora, including SONICOM's 793-direction protocol [23] and its 300-subject extension [24]. The simulation route replaces measurement with numerics: given a high-quality 3-D scan, boundary-element solvers such as Mesh2HRTF compute listener-specific HRTFs [6, 7], and P-HRTF demonstrated the full photos → photogrammetry → simulation pipeline on consumer cameras [8]. These methods are physically grounded but computationally heavy — Zhou *et al.* contrast 20–30 min of numerical simulation per subject with milliseconds for a learned predictor [13] — and remain sensitive to mesh quality. Our approach keeps the camera and drops the physics engine.

**Learning HRTFs from anthropometrics and images.** Zotkin *et al.* pioneered database matching from photo-derived pinna measurements [4]; later regressors map anthropometric vectors to HRTFs or their spherical-harmonic coefficients [14]. Images entered the pipeline first as an intermediate step — DeepEarNet regresses ear-shape-model parameters from stereo photos, then still simulates acoustically [9] — and later as a direct input alongside anthropometrics: Lee & Kim fuse an ear-image CNN with an anthropometric DNN on CIPIC [10], Zhao *et al.* combine VGG ear features with measured dimensions to predict spherical-harmonic magnitudes [15], and Miccini & Spagnol learn autoencoder latents of pinna images and depth maps for pinna-related transfer-function synthesis [11, 12]. Perceptual-feedback [16] and in-the-wild acoustic [17] calibration offer measurement-free alternatives that require listening sessions or instrumented earbuds. Relative to these, our conditioning signal is photographs alone — no anthropometric measurements, no intermediate shape model, no listening test — and the output is the complete binaural, full-sphere impulse-response tensor at native challenge resolution. This configuration was set by the challenge's input specification rather than chosen as a research position, and we make no priority claim relative to prior image-conditioned work.

**Neural HRTF representations.** A parallel literature learns *representations of the HRTF itself*: neural fields over direction with per-subject latents [18], IIR-filter fields [19], retrieval-augmented fields [20], and GAN [21] and diffusion [22] synthesis — largely aimed at spatial upsampling and dataset harmonization, as in the SONICOM/IEEE LAP Challenge [25], where personalization is conditioned on *acoustic* measurements of the listener. Our problem is complementary: the conditioning signal is purely visual, and the decoder produces the full sphere at once rather than answering per-direction queries.

**Multi-view aggregation.** Aggregating CNN features across views is classical in 3-D shape recognition — via order-invariant pooling (MVCNN [32]), learned set functions [34], or recurrent fusion (Ma *et al.* [33]). Our cross-view BiLSTM follows the recurrent line, a natural fit here because the challenge's pinna views follow a nominally consistent capture order — a sequence, not an unordered set — while tolerating any sequence *length*.

## 3. Method

<p align="center">
  <img src="assets/fig2_architecture.svg" width="100%" alt="Architecture: ResNet-50 per-view encoder, cross-view BiLSTM, spatial-expansion BiLSTM, shared MLP head, MSD loss."/>
</p>

<p align="center"><sub><b>Figure 2.</b> Architecture. A weight-shared, grayscale-adapted ResNet-50 embeds each of the 2<i>V</i> pinna views; a 4-layer bidirectional LSTM fuses the variable-length view sequence into a 2048-dim identity code <b>z</b>; a second bidirectional LSTM expands <b>z</b> into 1,586 output descriptors; and a shared MLP decodes each descriptor into a 256-tap head-related impulse response. Training minimizes the critical-band-weighted Mean Spectral Distortion — the challenge's evaluation metric — end-to-end.</sub></p>

### 3.1 Problem formulation

Given $V$ grayscale photographs of each ear, $\mathbf{X} \in \mathbb{R}^{2 \times V \times 1 \times 256 \times 256}$ with $`V \in \{19, 7, 3\}`$, predict the listener's head-related impulse responses $\hat{\mathbf{h}} \in \mathbb{R}^{793 \times 2 \times 256}$ over the full SONICOM measurement sphere (793 directions [23]) for both ears at 48 kHz. Supervision compares **magnitude spectra only**: the ground-truth SOFA files (`FreeFieldCompMinPhase`, ITD-removed variant) are transformed by a 256-point real FFT, and the predicted taps are transformed identically. The phase of the predicted impulse responses is therefore unconstrained — a limitation we return to in §5.

### 3.2 Multi-view pinna encoder

Each view is embedded independently by a ResNet-50 [29] pretrained on ImageNet. Two surgical changes adapt it to this domain: the stem convolution is replaced by a freshly Kaiming-initialized single-channel counterpart, and the classification head is removed in favor of global average pooling, yielding one 2048-dim descriptor per view. All $2V$ views — left and right ear alike — share the same encoder weights, so the encoder's job is purely *"describe this pinna"*; lateralization is handled downstream. With only 90 training subjects, the pretrained mid- and high-level features are retained while only the stem is relearned; we expect this to matter more than training a 25M-parameter backbone from scratch would allow, but we did not run the from-scratch control.

### 3.3 Variable-cardinality cross-view aggregation

The $2V$ view descriptors form a sequence that a 4-layer bidirectional LSTM (hidden size 1024, dropout 0.3) consumes; the final forward and backward hidden states are concatenated into a single **identity code** $\mathbf{z} \in \mathbb{R}^{2048}$. Because the summary state exists for *any* sequence length, the identical trained network ingests 38, 14, or 6 images without architectural change — one checkpoint serves all three challenge tasks. Bidirectionality lets every view's contribution be informed by the full set (e.g., resolving pose ambiguity of one crop from its neighbors). The recurrent design can in principle exploit the capture order of the challenge views rather than discarding it, though we did not test this against an order-invariant aggregator such as mean-pooling or a Set Transformer [34].

### 3.4 Recurrent spatial-expansion decoder

The decoder must inflate one 2048-dim code into $793 \times 2 = 1{,}586$ channels of 256 taps each — 406,016 output values, a ×198 expansion. We treat this generatively: $\mathbf{z}$ is unrolled as a 2048-step scalar sequence and driven through a 2-layer bidirectional LSTM of hidden size 793, whose output tensor (2048 steps × 1,586 bidirectional units) is reshaped to $1{,}586 \times 2048$ and decoded row-wise by a shared MLP ($2048 \to 512 \to 256$, ReLU, dropout 0.3) into 256-tap impulse responses.

Two honest caveats about this design. First, the reshape is a flat `view`, not a transpose: a row of the resulting grid does not correspond to a single channel of the recurrent output, so the mapping from recurrent state to output direction is *learned* rather than imposed by the architecture. Second, our motivation was primarily parameter efficiency — the expansion LSTM and head together use ≈21M parameters, against ≈832M for the dense head ($2048 \to 1{,}586 \times 256$) that the same job would otherwise require. It may additionally impose an implicit smoothness prior through shared recurrent dynamics, but the output channels are ordered by SOFA measurement index rather than by spatial adjacency, and we ran no ablation to test this.

### 3.5 Perceptually weighted training objective

<p align="center">
  <img src="assets/fig3_objective.svg" width="88%" alt="Critical-band frequency weighting curve and the ±30° elevation evaluation band."/>
</p>

<p align="center"><sub><b>Figure 3.</b> The training objective <i>is</i> the evaluation metric. (a) Frequency bins are weighted inversely to the auditory critical bandwidth (Zwicker–Terhardt approximation [27]), following the HRTF-similarity criterion of Bondu, Busson, Lemaire and Nicol [26]. (b) The error is averaged over the ±30° elevation band specified by the challenge protocol.</sub></p>

The challenge scores submissions with the **Mean Spectral Distortion (MSD)** — the critical-band mean squared error of the challenge rules:

```math
\mathrm{MSD} = 10 \log_{10} \Big( \mathbb{E}_{\,d \in \mathcal{D},\, e,\, k} \big[ \big( w_k \, ( |H_{d,e,k}| - |\hat{H}_{d,e,k}| ) \big)^2 \big] \Big) \ \text{dB},
```

where $\mathcal{D}$ is the set of directions with elevation in $[-30^\circ, +30^\circ]$, $k$ indexes the 129 rFFT bins spanning 0–24 kHz, and the weights $w_k \propto 1/\Delta_{\mathrm{cb}}(f_k)$ (normalized to sum to 1) are the inverse of the critical bandwidth $`\Delta_{\mathrm{cb}} = 25 + 75\,(1 + 1.4 f_{\mathrm{kHz}}^2)^{0.69}`$ Hz [26, 27].

The challenge distributed a reference implementation of this metric for scoring (`metrics.py`, part of the starter kit). Because it is already written in differentiable PyTorch operations, we use it **unchanged as the training loss** rather than optimizing a proxy (plain MSE on spectra or taps) and hoping it correlates with the score — so every gradient step is an explicit descent step on the leaderboard criterion. Two details are worth stating plainly. First, we keep the organizers' definition verbatim, weight *inside* the square; this is a squared-weight variant of the classical critical-band mean squared error [26], which weights squared errors directly. Since submissions were ranked by this exact expression, we optimize it as specified rather than the textbook form. Second, because the error is a difference of linear-scale magnitudes wrapped in a log, MSD is not commensurable with the log-spectral distortion values reported elsewhere in the literature (§4.3).

### 3.6 Subject-consistent augmentation

With 90 training subjects, augmentation is essential — but naive per-image augmentation would present the aggregator with views whose relative geometry and photometry are mutually inconsistent, corrupting exactly the cross-view signal §3.3 relies on. Our `ConsistentTransform` therefore *decouples parameter sampling from parameter application*: one call to `generate_params()` draws a single augmentation state — rotation (±10°), scale (0.9–1.1), translation (±10%), brightness/contrast (0.9–1.1) — which is then applied **identically to every view** of a subject until it is redrawn. Re-drawing once per subject presents the model with a new "virtual subject" while leaving the within-subject view relationships intact.

Two caveats for anyone reading the code: the transform caches its parameters and relies on the training loop to call `generate_params()` per subject, and that loop is not included in this repository, so with the code as shipped the state is drawn once on first use. The sampled Gaussian-blur sigma (0.1–0.2 on a 3×3 kernel) is numerically close to identity and is best regarded as inactive.

## 4. Experiments

### 4.1 Dataset and protocol

We use the challenge release of the **SONICOM HRTF dataset** [23] (Audio Experience Design group, Imperial College London; EU H2020 SONICOM project): 100 subjects — 90 for training, 10 held out for evaluation — each with measured HRTFs at 793 directions per ear (SOFA, `FreeFieldCompMinPhase` variant with ITDs removed [23, 28]) at 48 kHz, paired with grayscale pinna crops (resized to 256 × 256 in our pipeline) derived from the dataset's around-the-head captures. Evaluation spans three view-sparsity regimes:

| Task | Views per ear | Total input images | Scenario |
|:----:|:---:|:---:|:---|
| 0 | 19 | 38 | Full capture sweep |
| 1 | 7 | 14 | Reduced sweep (every 3rd view) |
| 2 | 3 | 6 | Minimal capture — three views per ear |

A single trained checkpoint is evaluated on all three tasks (§3.3); no per-task finetuning is performed.

### 4.2 Implementation details

The encoder is ResNet-50 (ImageNet weights, grayscale stem, headless); the aggregator a 4-layer BiLSTM (hidden 1024, dropout 0.3); the expansion decoder a 2-layer BiLSTM (hidden 793, dropout 0.3); the head an MLP 2048→512→256. Inputs are normalized to mean 0.5, std 0.5. Training minimizes the differentiable MSD of §3.5 end-to-end (including the pretrained encoder) with subject-consistent augmentation (§3.6); model selection keeps the checkpoint with the best evaluation MSD.

*Reproducibility.* The training hyperparameters (optimizer, learning-rate schedule, batch size, epoch count, hardware) were not recorded at submission time and are not recoverable from this repository; the reported figure comes from a single run with no fixed seed and no seed-variance estimate. Model selection and the reported score both used the 10 evaluation subjects described above, so **−63.6 dB is a best-checkpoint figure rather than a clean held-out estimate** and should be read as optimistic. We flag this explicitly rather than presenting the number as an unbiased generalization estimate.

Inference for a full 793 × 2 × 256 HRIR tensor is a single forward pass, and predictions are exported as SOFA files [28].

### 4.3 Results

| | Mean Spectral Distortion (dB, more negative is better) |
|:---|:---:|
| **Ours (submitted model)** | **−63.6** |

−63.6 dB corresponds to a weighted mean squared magnitude error of ≈4.4 × 10⁻⁷ on linear-scale magnitudes. The submission earned a **2nd prize** at Huawei's 2024 Munich Tech Arena.

*What this table does not contain.* Per-task (19/7/3-view) scores were not retained from the submission run, so the aggregate cannot be decomposed by regime. We also did not retain the challenge's population-average-HRTF baseline score on our evaluation split, and we do not have other teams' scores, so this figure stands without a quantitative comparison — the placement is the only external reference point we can offer. `plot_results.py` overlays a predicted magnitude response on the measured one for a test subject, which is the qualitative check we used during development; note that it averages across directions, which removes exactly the direction-dependent spectral detail discussed in §1, so it is not diagnostic for pinna-notch fidelity.

*A note on comparability.* MSD as defined in §3.5 operates on **linear-scale magnitude differences** inside a log; its dB values are therefore on a completely different scale from the log-spectral-distortion (LSD) numbers — typically 3–6 dB — reported in the personalization literature [10, 14, 15]. The two must not be compared directly.

## 5. Discussion and Limitations

**What the result suggests.** Sparse, uncalibrated 2-D photographs carry enough information to regress a perceptually weighted approximation of an individual's full-sphere HRTF magnitude, without an explicit 3-D intermediate — consistent with Zhou *et al.*'s finding that deep networks can predict much of an individual's HRTF from ear geometry [13], though here from raw pixels rather than 3-D shape. The variable-cardinality design is the practically important part: a deployed system cannot dictate how many usable photos a user provides.

**Limitations.**

1. *Evaluation rigor.* One aggregate number, one run, no seed variance, no ablations, and a checkpoint selected on the same 10 subjects it is scored on (§4.2). None of the architectural choices in §3 is supported by a controlled comparison.
2. *Data scale.* 90 training subjects is small; the public SONICOM corpus has since grown to 300 measured subjects [24], and scaling studies are the obvious next step.
3. *Magnitude-only supervision.* Supervision constrains magnitude spectra alone, on an ITD-removed target variant. The predicted impulse responses therefore carry no calibrated phase and no interaural time structure: the exported SOFA files are format-conformant but require ITD re-injection (e.g. from a lightweight head-size estimate, as the SONICOM ear-aligned format anticipates [23]) and a phase model before binaural rendering.
4. *Metric–motivation mismatch.* The challenge's inverse-critical-bandwidth weighting places ≈74% of its mass below 5 kHz and only ≈13% in the 5–10 kHz band where the elevation-defining pinna notches live (the DC weight is 17× the 8 kHz weight). A model trained directly on this metric is thus optimized *least* where the Introduction's motivating cues are strongest; a notch-aware or high-frequency-reweighted objective is an obvious follow-up.
5. *Metric vs. perception.* MSD is perceptually weighted but is not a listening test; localization studies with rendered audio are needed for any perceptual claim.
6. *Architectural priors.* The sequence-unrolling decoder imposes no explicit spatial structure (§3.4); geometry-aware decoders on sphere-respecting representations — equiangular projections [21] or spherical-harmonic coefficients [14, 15] — and attention-based set aggregation [34] are natural upgrades.
7. *Capture conditions.* All inputs come from SONICOM's controlled around-the-head capture rig. Generalization to consumer phone imagery, uncontrolled lighting, and hair occlusion is untested.

## 6. Conclusion

We presented a single end-to-end network that turns a handful of ear photographs into a complete personalized HRTF magnitude set in SOFA format — handling 19, 7, or 3 views per ear with one set of weights, decoding all 1,586 output channels jointly, and training directly on the perceptually weighted metric it is evaluated with. The system took a 2nd prize at Huawei's Munich Tech Arena. We see it as a promising sign that HRTF personalization can shed its remaining acquisition hardware, with the caveat that all inputs here come from a controlled capture rig; whether ordinary phone imagery supports the same mapping is untested.

---

## Repository

```
├── assets/               # Paper figures (SVG)
├── model.py              # ResNetEncoder + HRTFModel (Fig. 2)
├── metrics.py            # Differentiable Mean Spectral Distortion (loss & eval, Fig. 3)
├── transformations.py    # Subject-consistent augmentation (§3.6)
├── utils.py              # SONICOM dataset class, SOFA I/O
├── inference.py          # CLI inference → SOFA; evaluate() for Tasks 0/1/2
├── plot_results.py       # Prediction vs. ground-truth spectra overlays
├── requirements.txt
└── LICENSE
```

**Provenance.** `metrics.py`, `utils.py`, and the CLI scaffold of `inference.py` are adapted from the challenge starter kit supplied by the organizers. `model.py`, `transformations.py`, the training loop, and the decision to train directly on the provided MSD metric are ours.

**What is not here.** The training loop ran in the challenge compute environment and is not included; the trained checkpoint (`best_model_vf5c.pth`, expected at the repo root by `inference.py`) is not distributed — open an issue if you would like it; and the SONICOM data is distributed by the [SONICOM project](https://www.sonicom.eu/tools-and-resources/hrtf-dataset/), not redistributed here. The code below therefore defines the architecture and objective but is not a turnkey reproduction package: the training schedule would have to be reconstructed.

### Setup

```bash
git clone https://github.com/clmrie/HRTF_prediction_Huawei.git
cd HRTF_prediction_Huawei
pip install -r requirements.txt
```

Expected data layout (SONICOM challenge release):

```
data/
├── Average_HRTFs.sofa                    # challenge-provided average-HRTF template
├── P0001/P0001/HRTF/HRTF/48kHz/*.sofa    # per-subject SOFA (FreeFieldCompMinPhase, NoITD)
├── SONICOM_TrainingData_pics/            # training pinna images (P####_left|right_##.png)
└── SONICOM_TestData_pics/                # test pinna images
```

### Inference

Predict a personalized HRTF from photographs of both ears (any view count; 19/7/3 correspond to the challenge tasks):

```bash
python inference.py \
    -l left_01.png  left_02.png  left_03.png \
    -r right_01.png right_02.png right_03.png \
    -o my_hrtf.sofa
```

| Flag | Meaning |
|:---|:---|
| `-l, --left` | Left-ear pinna image paths (space-separated) |
| `-r, --right` | Right-ear pinna image paths (space-separated) |
| `-o, --output_path` | Output SOFA file (793 directions × 2 ears × 256-tap HRIRs) |

### Evaluation & visualization

```bash
# MSD on the evaluation split for all three tasks:
# in inference.py's __main__, call evaluate() instead of main(), then
python inference.py

# Overlay predicted vs. measured magnitude spectra for a subject:
python plot_results.py -s P0004 -p predicted_hrtf_P0004.sofa -t 2
```

## References

<sub>

[1] J. Blauert. *Spatial Hearing: The Psychophysics of Human Sound Localization.* MIT Press, revised ed., 1997.<br>
[2] D. W. Batteau. The role of the pinna in human localization. *Proc. Royal Society B*, 168(1011):158–180, 1967.<br>
[3] M. B. Gardner and R. S. Gardner. Problem of localization in the median plane: effect of pinnae cavity occlusion. *J. Acoust. Soc. Am.*, 53(2):400–408, 1973.<br>
[4] D. N. Zotkin, J. Hwang, R. Duraiswami, and L. S. Davis. HRTF personalization using anthropometric measurements. *IEEE WASPAA*, 2003.<br>
[5] C. Guezenoc and R. Séguier. HRTF individualization: A survey. *AES 145th Convention*, paper 10129, 2018.<br>
[6] H. Ziegelwanger, P. Majdak, and W. Kreuzer. Numerical calculation of listener-specific head-related transfer functions and sound localization. *J. Acoust. Soc. Am.*, 138(1):208–222, 2015.<br>
[7] H. Ziegelwanger, W. Kreuzer, and P. Majdak. Mesh2HRTF: An open-source software package for the numerical calculation of head-related transfer functions. *ICSV22*, 2015.<br>
[8] A. Meshram, R. Mehra, H. Yang, E. Dunn, J.-M. Frahm, and D. Manocha. P-HRTF: Efficient personalized HRTF computation for high-fidelity spatial sound. *IEEE ISMAR*, pp. 53–61, 2014.<br>
[9] S. Kaneko, T. Suenaga, and S. Sekine. DeepEarNet: Individualizing spatial audio with photography, ear shape modeling, and neural networks. *AES Int. Conf. on Audio for Virtual and Augmented Reality*, 2016.<br>
[10] G. W. Lee and H. K. Kim. Personalized HRTF modeling based on deep neural network using anthropometric measurements and images of the ear. *Applied Sciences*, 8(11):2180, 2018.<br>
[11] R. Miccini and S. Spagnol. HRTF individualization using deep learning. *IEEE VRW*, pp. 390–395, 2020.<br>
[12] R. Miccini and S. Spagnol. A hybrid approach to structural modeling of individualized HRTFs. *IEEE VRW*, 2021.<br>
[13] Y. Zhou, H. Jiang, and V. K. Ithapu. On the predictability of HRTFs from ear shapes using deep networks. *IEEE ICASSP*, 2021.<br>
[14] Y. Wang, Y. Zhang, Z. Duan, and M. Bocko. Global HRTF personalization using anthropometric measures. *AES 150th Convention*, paper 10502, 2021.<br>
[15] M. Zhao, Z. Sheng, and Y. Fang. Magnitude modeling of personalized HRTF based on ear images and anthropometric measurements. *Applied Sciences*, 12(16):8155, 2022.<br>
[16] K. Yamamoto and T. Igarashi. Fully perceptual-based 3D spatial sound individualization with an adaptive variational autoencoder. *ACM Trans. Graphics (SIGGRAPH Asia)*, 36(6), 2017.<br>
[17] V. Jayaram, I. Kemelmacher-Shlizerman, and S. M. Seitz. HRTF estimation in the wild. *ACM UIST*, 2023.<br>
[18] Y. Zhang, Y. Wang, and Z. Duan. HRTF Field: Unifying measured HRTF magnitude representation with neural fields. *IEEE ICASSP*, 2023.<br>
[19] Y. Masuyama, G. Wichern, F. G. Germain, Z. Pan, S. Khurana, C. Hori, and J. Le Roux. NIIRF: Neural IIR filter field for HRTF upsampling and personalization. *IEEE ICASSP*, pp. 1016–1020, 2024.<br>
[20] Y. Masuyama, G. Wichern, F. G. Germain, C. Ick, and J. Le Roux. RANF: Retrieval-augmented neural field for HRTF upsampling and personalization. *IEEE ICASSP*, 2025.<br>
[21] A. O. T. Hogg, M. Jenkins, H. Liu, I. Squires, S. J. Cooper, and L. Picinali. HRTF upsampling with a generative adversarial network using a gnomonic equiangular projection. *IEEE/ACM Trans. Audio, Speech, Lang. Process.*, 2024.<br>
[22] J. C. Albarracín Sánchez, L. Comanducci, M. Pezzoli, and F. Antonacci. Towards HRTF personalization using denoising diffusion models. *IEEE ICASSP*, 2025.<br>
[23] I. Engel, R. Daugintis, T. Vicente, A. O. T. Hogg, J. Pauwels, A. J. Tournier, and L. Picinali. The SONICOM HRTF dataset. *J. Audio Eng. Soc.*, 71(5):241–253, 2023. DOI 10.17743/jaes.2022.0066.<br>
[24] K. C. Poole, J. Meyer, V. Martin, R. Daugintis, N. Marggraf-Turley, J. Webb, L. Pirard, N. La Magna, O. Turvey, and L. Picinali. The extended SONICOM HRTF dataset and spatial audio metrics toolbox. *Forum Acusticum*, 2025. arXiv:2507.05053.<br>
[25] R. Daugintis, A. O. T. Hogg, L. Picinali, et al. Technical report: SONICOM / IEEE Listener Acoustic Personalisation (LAP) Challenge 2024. TechRxiv, 2024. DOI 10.36227/techrxiv.173153187.72930965.<br>
[26] A. Bondu, S. Busson, V. Lemaire, and R. Nicol. Looking for a relevant similarity criterion for HRTF clustering: a comparative study. *AES 120th Convention*, paper 6653, 2006.<br>
[27] E. Zwicker and E. Terhardt. Analytical expressions for critical-band rate and critical bandwidth as a function of frequency. *J. Acoust. Soc. Am.*, 68(5):1523–1525, 1980.<br>
[28] P. Majdak, Y. Iwaya, T. Carpentier, R. Nicol, M. Parmentier, A. Roginska, Y. Suzuki, K. Watanabe, H. Wierstorf, H. Ziegelwanger, and M. Noisternig. Spatially Oriented Format for Acoustics: A data exchange format representing head-related transfer functions. *AES 134th Convention*, paper 8880, 2013. Standardized as AES69-2015.<br>
[29] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. *IEEE CVPR*, 2016.<br>
[30] S. Hochreiter and J. Schmidhuber. Long short-term memory. *Neural Computation*, 9(8):1735–1780, 1997.<br>
[31] M. Schuster and K. K. Paliwal. Bidirectional recurrent neural networks. *IEEE Trans. Signal Processing*, 45(11):2673–2681, 1997.<br>
[32] H. Su, S. Maji, E. Kalogerakis, and E. Learned-Miller. Multi-view convolutional neural networks for 3D shape recognition. *IEEE ICCV*, 2015.<br>
[33] C. Ma, Y. Guo, J. Yang, and W. An. Learning multi-view representation with LSTM for 3-D shape recognition and retrieval. *IEEE Trans. Multimedia*, 21(5):1169–1182, 2019.<br>
[34] J. Lee, Y. Lee, J. Kim, A. R. Kosiorek, S. Choi, and Y. W. Teh. Set Transformer: A framework for attention-based permutation-invariant neural networks. *ICML*, 2019.<br>

</sub>

## Acknowledgments

Built for the **2024 Munich Tech Arena** — Huawei's Media Technology Competition (finals in Munich, Germany) — on the **SONICOM HRTF dataset** [23] created by the Audio Experience Design group at Imperial College London (SONICOM is funded by the EU Horizon 2020 programme, grant No. 101017743). We thank Huawei for organizing the challenge and supplying the data release, evaluation protocol, and starter kit, and the SONICOM consortium for making the dataset openly available.

## License & Contact

Released under the [MIT License](LICENSE).
Questions: alakbarlirasul@gmail.com · or open an issue.
