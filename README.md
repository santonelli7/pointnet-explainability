# PointNet Explanation
Model-level technique to explain what PointNet, trained on 3D point cloud classification, has learned.

### Requirements
`pip install -r requirements.txt`

## Train PointNet
1. `python -m scripts.train_pointnet -c "settings/pointnet_hyperparams.json"`

## Train AAE
2. `python -m scripts.train_aae -c "./settings/aae_hyperparams.json"`

## Input Optimization
3. `python -m scripts.optimize -c "settings/opt_hyperparams.json"`

## Resources
- [PointNet](https://github.com/fxia22/pointnet.pytorch) implementation
- [Adversarial Autoencoder](https://github.com/MaciejZamorski/3d-AAE) implementation