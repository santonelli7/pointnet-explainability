# PointNet Explanation
Model-level technique to explain what PointNet learnt.

### Train PointNet
1. `python3 -m tools/train_pointnet -c "settings/pointnet_hyperparams.json"`

### Train AAE
2. `python3 -m tools/train_aae -c "./settings/aae_hyperparams.json"`

### Input Optimization
3. `python -m tools/optimize -c "settings/opt_hyperparams.json"`

# Resources
- [PointNet](https://github.com/fxia22/pointnet.pytorch) implementation
- [Adversarial Autoencoder](https://github.com/MaciejZamorski/3d-AAE) implementation