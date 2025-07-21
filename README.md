# Schwarz-Schur Involution or Sparse Inversion

This repo is the official implementation of the ICML 2025 paper 

*    **Schwarz–Schur Involution: Lightspeed Differentiable Sparse Linear Solvers**
  
        by *Yu Wang, S. Mazdak Abulnaga, Yaël Balbastre, Bruce Fischl.*
     
        *International Conference on Machine Learning (ICML) 2025.*

<img width="3200" height="1600" alt="poster-icml25" src="https://github.com/user-attachments/assets/0249964a-cb50-47cb-b07c-2c8158dfc180" />

(Title in Chinese: Schwarz-Schur翻卷, 稀疏矩阵逆)

[OpenReview](https://openreview.net/pdf?id=RKbanvzycr)

[Poster](https://www.dropbox.com/scl/fi/l3d7y338wk1nthm9zw3cy/poster-icml25.pdf?rlkey=56rf2a9ft68sitptdsx5280y7&st=h3sb7yxz&dl=0)

# Usage

The basic usage: 

```
import torch_sparse_involution as tsi
sol = tsi.Schwarz_Schur_involution(alpha, beta, BC=BC, wh=ab_wh[1])
```

solves the linear system Ax=b, by storing the linear system `A, b` in the pair of tensors `alpha, beta`,  
after preparing the input tensors alpha, beta, boundary condition, and hyper-parameters. 

For example, 
```
ab_wh=((128,128), (5,5))
alpha = ... # [batch_size, a, b, (w*h), (w*h)]
beta  = ... # [batch_size, a, b, (w*h), channels]
BC = 'Neumann'
```

More details are coming soon. 

# Demo, Tutorial

Coming soon after ICML. 

# Reference 

```
@inproceedings{
wang2025schwarzschur,
title={Schwarz{\textendash}Schur Involution: Lightspeed Differentiable Sparse Linear Solvers},
author={Yu Wang and Mazdak Abulnaga and Ya{\"e}l Balbastre and Bruce Fischl},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://github.com/wangyu9/Schwarz_Schur_Involution}
}
```
