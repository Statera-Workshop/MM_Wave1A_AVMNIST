# Third-party code and data

| Component | Source | Licence | Used for |
| --- | --- | --- | --- |
| LeNet, MLP, Concat, TensorFusion, LowRankTensorFusion, AV-MNIST loader layout | [MultiBench](https://github.com/pliang279/MultiBench) | MIT (`licenses/MultiBench_LICENSE.txt`) | encoders and fusion modules in `code/mm_a.py`; LowRankTensorFusion re-implemented with registered parameters |
| OGM-GE modulation (score ratio, coefficient, GE noise) | [GeWu-Lab/OGM-GE_CVPR2022](https://github.com/GeWu-Lab/OGM-GE_CVPR2022) | MIT (`licenses/OGM-GE_LICENSE.txt`) | `ogm_coeffs` / `train_step` in `code/mm_a.py` |
| Gradient Blending | paper: Wang, Tran, Feiszli, *What Makes Training Multi-modal Classification Networks Hard?*, CVPR 2020 | - | implemented from the paper |

**Data (not redistributed).** AV-MNIST as released by MultiBench: MNIST digits (CC BY-SA 3.0), Free Spoken Digit
Dataset recordings (CC BY-SA 4.0) and ESC-50 background noise (**CC BY-NC 3.0**). Because of ESC-50 the released
weights are for non-commercial use.
