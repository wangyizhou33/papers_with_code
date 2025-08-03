## Definitions 

Cosmos-Drive-Dreams = a synthetic data generation (SDG) pipeline that aims to generate scenarios.  
Cosmos-Drive = a suite of models specialized from NVIDIA Cosmos-1 world foundation model. 

- Cosmos-7B-AV-Sample (Paper Sec. [2.1]).  `text-to-videomodel`
- Cosmos-7B-Multiview-AV-Sample (Paper Sec. [2.1]).   `text-to-videomodel`
- Cosmos-Transfer1-7B-Sample-AV (Paper Sec. [2.2]).   `dense conditional models`
- Cosmos-7B-Single2Multiview-Sample-AV (Paper Sec. [2.3]).  `dense conditional models`
- Cosmos-7B-Annotate-Sample-AV (Paper Sec. [2.4]).  `dense conditional models`
- Cosmos-7B-LiDAR-GEN-Sample-AV (Paper Sec. [3]).   

## Usages
- Diverse Generation from Cosmos-Drive
- Multi-View Generation from Cosmos-Drive
- Corner Case Generation from Cosmos-Drive
- LiDAR Generation from Cosmos-Drive

## Cosmos-1 WFM
- Cosmos-7B-Text2World: text-to-video diffusion model
- Cosmos-Transfer1: incorporating multiple ControlNets [68], including segmentation map control, Canny edge control, depth control, and blur video
control.


## Model lineage
`Cosmos-7B-Text2World` -> RDS -> `Cosmos-7B-AV-Sample` -> controlnets, RDS-HQ -> `Cosmos-Transfer1-7B-Sample-AV`.  
`Cosmos-7B-Text2World` -> RDS -> `Cosmos-7B-Multiview-Sample-AV`  -> controlnets, RDS-HQ -> `Cosmos-7B-Single2Multiview-Sample-AV`.  