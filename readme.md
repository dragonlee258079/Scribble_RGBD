Code release for the TPAMI 2023 paper "Robust Perception and Precise Segmentation for Scribble-Supervised RGB-D Saliency Detection" by Long Li, Junwei Han, Nian Liu*, Salman Khan, Hisham Cholakkal, Rao Muhammad Anwer, and Fahad Shahbaz Khan.

![avatar](framework.jpg)

## Code Construction Explanation
This paper involves three main models: our FPN-based (**Ours_FPN**) and HRNet-based (**Ours_HRNet**) models trained on our NNDR dataset, as well as our FPN-based model (**Ours_FPN_Xu**) trained on Xu's dataset. These three models share a common framework but have different hyperparameters. Thus, we prepared three separate projects for each model in this repository to facilitate  the use of our code. 

## Abstract
This paper proposes a scribble-based weakly supervised RGB-D salient object detection (SOD) method to relieve the annotation burden from pixel-wise annotations. In view of
the ensuing performance drop, we summarize two natural deficiencies of the scribbles and try to alleviate them, which are the weak richness of the pixel training samples (WRPS) and the poor structural integrity of the salient objects (PSIO). WRPS hinders robust saliency perception learning, which can be alleviated via model design for robust feature learning and pseudo-label generation for training sample enrichment. Specifically, we first design a dynamic searching process module as a meta operation to conduct multi-scale and multi-modal feature fusion for the robust RGB-D SOD model construction. Then, a dual-branch consistency learning mechanism is proposed to generate enough pixel training samples for robust saliency perception learning. PSIO makes direct structural learning infeasible since scribbles can not provide integral structural supervision. Thus, we propose an edge-region structure-refinement loss to recover the structural information and make precise segmentation. We deploy all components and conduct ablation studies on two baselines to validate their effectiveness and generalizability. Experimental results on eight datasets show that our method outperforms other scribble-based SOD models and achieves comparable performance with fully supervised state-of-the-art methods.

## Environment Configuration
- Linux with Python ≥ 3.6
- PyTorch ≥ 1.7 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`


## Result
You can download the saliency maps of our models, which were trained on both our dataset (`Ours_FPN`, `Ours_HRNet`) and Xu's dataset (`Ours_FPN_Xu`), as well as other models that were trained on our dataset (WSSA, SCWS, DENet-N), from [saliency maps](https://drive.google.com/drive/folders/1mpPVDuke88qxtuC47OgLCsLwpx-gqSKK?usp=sharing).

![alt_text](./result_quantitation.jpg)
![alt_text](./result_qualitative.jpg)
