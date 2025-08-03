
## Survey 

- digital twins for testing autonomous systems [8, 24, 52]
- generate diverse data for training end-to-end planning algorithms [2, 9, 12].

- driving scene reconstruction
  - nerf: 27, 35, 39, 47
  - 3dgs: 17, 54, 53, ...
  
- novel-view synthesis 
  - regularization-guided 18,41
  - generative-prior-guided (diffusion) 14, 46, 49

- appearance modeling
- multi-traversal street reconstruction


## Contributions

The contributions are summarized as follows:
- We propose MTGS with a novel multi-traversal scene
graph, including a shared static node that represents back-
ground geometry, an appearance node to model various
appearances, and a transient node to preserve dynamic in-
formation.  

- MTGS enables high-fidelity reconstruction with extraor-
dinary view extrapolation quality. We demonstrate that
the MTGS achieves state-of-the-art performance in driv-
ing scene extrapolated view synthesis. It outperforms pre-
vious SOTA by 17.6% on SSIM, 42.4% on LPIPS and
35% on AbsRel.

