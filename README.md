
<div align="center">    
 
# Uncovering the Folding Landscape of RNA Secondary Structure with Deep Graph Embeddings
<!-- 
[![Paper](http://img.shields.io/badge/paper-arxiv.2006.06885.svg)](https://arxiv.org/abs/2006.06885)

[![Conference](http://img.shields.io/badge/ICLR-GRL+-2020-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
 -->

[![Paper](https://img.shields.io/badge/arxiv-2006.06885-B31B1B.svg)](https://arxiv.org/abs/2006.06885)




<!--  
Conference   
-->   
</div>
 
## Description   

Biomolecular graph analysis has recently gained much attention in the emerging field of geometric deep learning. While numerous approaches aim to train classifiers that accurately predict molecular properties from graphs that encode their structure, an equally important task is to organize biomolecular graphs in ways that expose meaningful relations and variations between them. We propose a geometric scattering autoencoder (GSAE) network for learning such graph embeddings. Our embedding network first extracts rich graph features using the recently proposed geometric scattering transform. Then, it leverages a semi-supervised variational autoencoder to extract a low-dimensional embedding that retains the information in these features that enable prediction of molecular properties as well as characterize graphs. Our approach is based on the intuition that geometric scattering generates multi-resolution features with in-built invariance to deformations, but as they are unsupervised, these features may not be tuned for optimally capturing relevant domain-specific properties. We demonstrate the effectiveness of our approach to data exploration of RNA foldings. Like proteins, RNA molecules can fold to create low energy functional structures such as hairpins, but the landscape of possible folds and fold sequences are not well visualized by existing methods. We show that GSAE organizes RNA graphs both by structure and energy, accurately reflecting bistable RNA structures. Furthermore, it enables interpolation of embedded molecule sequences mimicking folding trajectories. Finally, using an auxiliary inverse-scattering model, we demonstrate our ability to generate synthetic RNA graphs along the trajectory thus providing hypothetical folding sequences for further analysis.

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/Your-project-name   

# install project   
cd Your-project-name 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to [Your Main Contribution (MNIST here)] and run it.   
 ```bash
# module folder
cd src/    

# run module (example: mnist as your main contribution)   
python simplest_mnist.py    
```

## Main Contribution      
List your modules here. Each module contains all code for a full system including how to run instructions.   
- [Production MNIST](https://github.com/PyTorchLightning/pytorch-lightning-conference-seed/tree/master/src/production_mnist)    
- [Research MNIST](https://github.com/PyTorchLightning/pytorch-lightning-conference-seed/tree/master/src/research_mnist)  

## Baselines    
List your baselines here.   
- [Research MNIST](https://github.com/PyTorchLightning/pytorch-lightning-conference-seed/tree/master/src/research_mnist) 

### Citation   
```
@article{castro2020uncovering,
  title={Uncovering the Folding Landscape of RNA Secondary Structure with Deep Graph Embeddings},
  author={Castro, Egbert and Benz, Andrew and Tong, Alexander and Wolf, Guy and Krishnaswamy, Smita},
  journal={arXiv preprint arXiv:2006.06885},
  year={2020}
}
```   
