# MM_Wave1A_AVMNIST

**Multimodal fusion convergence on AV-MNIST - PAI-SG Multimodal Wave 1A**

Part of the *Physical AI* reference series of the **Statera Working Group**. Convergence of multimodal fusion methods trained from scratch on AV-MNIST - how fast and how stably each fusion converges, and how much each modality is learned. Method repositories: [MM_Fusion_LateFusion](https://github.com/Statera-Workshop/MM_Fusion_LateFusion), [MM_Fusion_TFN](https://github.com/Statera-Workshop/MM_Fusion_TFN), [MM_Fusion_LMF](https://github.com/Statera-Workshop/MM_Fusion_LMF), [MM_Fusion_GradBlend](https://github.com/Statera-Workshop/MM_Fusion_GradBlend), [MM_Fusion_OGMGE](https://github.com/Statera-Workshop/MM_Fusion_OGMGE).

## Protocol

**Common setup** (identical for all AV-MNIST repositories of this wave).
AV-MNIST (MultiBench pre-processed release): 28 x 28 PCA-reduced MNIST digit + 112 x 112 spectrogram of a spoken digit
with ESC-50 background noise, 10 classes; train 55,000 / val 5,000 (last 5,000 of the training file) / test 10,000 pairs.
Encoders from scratch: image LeNet(1,6,3) -> 48-d, audio LeNet(1,6,5) -> 192-d (MultiBench); image standardised with
the training mean / std, audio / 255. Adam 1e-3 constant, batch 64, budgets 30 and 100 epochs, seeds 777 / 778 / 779,
fp32, deterministic cuDNN. Every epoch: test accuracy and a frozen linear probe per modality (10,000 training pairs,
standardised features, full-batch Adam, test accuracy). Reported value = last epoch.

**Convergence measures.** Peak test accuracy and its epoch, drop after the peak (peak - final), and the *audio gap* =
audio-only model's probe - this model's audio probe (same budget; positive = audio under-learned in the fusion model).
*BN re-estimated*: the saved last-epoch weights evaluated again after recomputing BatchNorm statistics on 200 training
batches (the series' evaluation practice); for AV-MNIST it changes nothing beyond +/-0.004.

## Results (AV-MNIST, mean of seeds; range in brackets)

| Model | Optimiser | Epochs | Seeds | Final test acc (range) | Peak (epoch) | Drop after peak | Audio probe | Audio gap | Final, BN re-estimated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image only | Adam 1e-3 | 30 | 3 | 0.650 (0.010) | 0.656 (20.7) | 0.006 | - | - | 0.651 (0.010) |
| image only | Adam 1e-3 | 100 | 3 | 0.644 (0.017) | 0.657 (40.7) | 0.013 | - | - | 0.646 (0.016) |
| audio only | Adam 1e-3 | 30 | 3 | 0.417 (0.001) | 0.424 (8.3) | 0.006 | 0.387 | - | 0.417 (0.001) |
| audio only | Adam 1e-3 | 100 | 3 | 0.419 (0.004) | 0.424 (33.0) | 0.006 | 0.385 | - | 0.419 (0.002) |
| Late Fusion (MLP head) | Adam 1e-3 | 30 | 3 | 0.696 (0.000) | 0.722 (8.3) | 0.027 | 0.359 | +0.028 | 0.696 (0.002) |
| Late Fusion (MLP head) | Adam 1e-3 | 100 | 3 | 0.679 (0.004) | 0.722 (8.3) | 0.044 | 0.362 | +0.024 | 0.678 (0.009) |
| Late Fusion (linear head) | Adam 1e-3 | 30 | 3 | 0.712 (0.008) | 0.723 (16.3) | 0.011 | 0.367 | +0.020 | 0.713 (0.008) |
| Late Fusion (linear head) | Adam 1e-3 | 100 | 3 | 0.708 (0.002) | 0.723 (16.3) | 0.015 | 0.380 | +0.005 | 0.708 (0.003) |
| Tensor Fusion Network | Adam 1e-3 | 30 | 3 | 0.668 (0.005) | 0.719 (5.7) | 0.051 | 0.380 | +0.007 | 0.669 (0.007) |
| Tensor Fusion Network | Adam 1e-3 | 100 | 3 | 0.670 (0.005) | 0.719 (5.7) | 0.049 | 0.389 | -0.004 | 0.671 (0.006) |
| Low-rank Multimodal Fusion | Adam 1e-3 | 30 | 3 | 0.679 (0.008) | 0.722 (7.0) | 0.043 | 0.370 | +0.016 | 0.680 (0.008) |
| Low-rank Multimodal Fusion | Adam 1e-3 | 100 | 3 | 0.678 (0.001) | 0.722 (7.0) | 0.044 | 0.371 | +0.015 | 0.678 (0.003) |
| Gradient Blending | Adam 1e-3 | 30 | 3 | 0.718 (0.006) | 0.729 (19.3) | 0.011 | 0.393 | -0.006 | 0.717 (0.005) |
| Gradient Blending | Adam 1e-3 | 100 | 3 | 0.691 (0.012) | 0.729 (19.3) | 0.038 | 0.393 | -0.007 | 0.691 (0.012) |
| OGM-GE | Adam 1e-3 | 30 | 3 | 0.712 (0.001) | 0.721 (14.0) | 0.009 | 0.353 | +0.034 | 0.712 (0.006) |
| OGM-GE | Adam 1e-3 | 100 | 3 | 0.709 (0.004) | 0.721 (14.0) | 0.012 | 0.366 | +0.020 | 0.710 (0.003) |
| OGM-GE | SGD (official) | 30 | 3 | 0.698 (0.006) | 0.707 (17.7) | 0.009 | 0.307 | +0.080 | 0.699 (0.009) |
| OGM-GE | SGD (official) | 100 | 3 | 0.701 (0.006) | 0.708 (34.0) | 0.007 | 0.304 | +0.081 | 0.701 (0.007) |
| OGM (no GE noise) | Adam 1e-3 | 30 | 3 | 0.711 (0.011) | 0.722 (11.7) | 0.011 | 0.376 | +0.010 | 0.712 (0.013) |
| OGM (no GE noise) | Adam 1e-3 | 100 | 3 | 0.708 (0.005) | 0.722 (22.3) | 0.014 | 0.380 | +0.005 | 0.710 (0.008) |

## Findings

- **Reached level is almost the same; the differences come after the peak.** All fusion models peak within about 0.01 of each other; the final differences are mostly the drop after the peak (overfitting).
- **Higher-capacity fusion peaks earlier and drops more** (TFN, LMF); linear-head fusion is the most stable.
- **Epoch-level speed (E90) carries no information here**: every model reaches 90 % of its final accuracy in the first epoch, so peak, peak epoch and drop are the convergence measures.
- **Modality imbalance is small and method-dependent.** The image is fully learned everywhere; Gradient Blending learns the audio best, OGM-GE worst - because of its GE noise (OGM without noise is at the control's level).
- MultiBench's LowRankTensorFusion does not train its fusion factors on GPU (fixed here, see MM_Fusion_LMF).

Full tables: `results/summary_A.md`; BN re-estimation of every saved model: `results/bn_reestimation_A.md`.

## Weights

Release **v1.0** of this repository (12 files, SHA-256 in `SHA256SUMS.txt`): `avmnist_aud_adam_e100_s777.pth`, `avmnist_aud_adam_e100_s778.pth`, `avmnist_aud_adam_e100_s779.pth`, `avmnist_aud_adam_e30_s777.pth`, `avmnist_aud_adam_e30_s778.pth`, `avmnist_aud_adam_e30_s779.pth`, `avmnist_img_adam_e100_s777.pth`, `avmnist_img_adam_e100_s778.pth`, `avmnist_img_adam_e100_s779.pth`, `avmnist_img_adam_e30_s777.pth`, `avmnist_img_adam_e30_s778.pth`, `avmnist_img_adam_e30_s779.pth`.

```python
import torch, sys; sys.path.insert(0, 'code'); import mm_a as M
m = M.AVNet('<model>')  # img, aud, late, late_lin, tfn, lmf, gblend, ogmge, ogm
m.load_state_dict(torch.load('avmnist_<model>_<opt>_e<epochs>_s<seed>.pth', weights_only=True), strict=True)
```

**Licence note.** The AV-MNIST audio contains ESC-50 noise (CC BY-NC 3.0), so the released weights are for non-commercial use. See THIRD_PARTY.md.

## Run

Data: `gdown 1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp -O avmnist.tar.gz` (MultiBench release), unpack to `~/MM_Wave1/data/avmnist`, check with `python code/count_avmnist.py ~/MM_Wave1/data/avmnist`.

```bash
cd code && bash smoke_a.sh && python make_queue_a.py
```
