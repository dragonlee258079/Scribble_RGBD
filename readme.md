Code release for the TPAMI 2023 paper "Robust Perception and Precise Segmentation for Scribble-Supervised RGB-D Saliency Detection" by Long Li, Junwei Han, Nian Liu*, Salman Khan, Hisham Cholakkal, Rao Muhammad Anwer, and Fahad Shahbaz Khan.

![avatar](framework.jpg)

## Abstract
This paper proposes a scribble-based weakly supervised RGB-D salient object detection (SOD) method to relieve the annotation burden from pixel-wise annotations. In view of
the ensuing performance drop, we summarize two natural deficiencies of the scribbles and try to alleviate them, which are the weak richness of the pixel training samples (WRPS) and the poor structural integrity of the salient objects (PSIO). WRPS hinders robust saliency perception learning, which can be alleviated via model design for robust feature learning and pseudo-label generation for training sample enrichment. Specifically, we first design a dynamic searching process module as a meta operation to conduct multi-scale and multi-modal feature fusion for the robust RGB-D SOD model construction. Then, a dual-branch consistency learning mechanism is proposed to generate enough pixel training
samples for robust saliency perception learning. PSIO makes direct structural learning infeasible since scribbles can not provide integral structural supervision. Thus, we propose an edge-region structure-refinement loss to recover the structural information and make precise segmentation. We deploy all components and conduct ablation studies on two baselines to validate their effectiveness and generalizability. Experimental results on eight datasets show that our method outperforms other scribble-based SOD models and
achieves comparable performance with fully supervised state-of-the-art methods.
