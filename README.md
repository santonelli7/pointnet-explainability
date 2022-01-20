# PointNet Explanation
Implementation of Input Optimization technique to explain the predictions of PointNet classifier. For a more detailed presentation of the approach, check <a href="https://santonelli7.github.io/projects/PointNetExpl" target="_blank">this</a> out.


### Requirements
`pip install -r requirements.txt`

## Train PointNet
1. `python -m scripts.train_pointnet -c "settings/pointnet_hyperparams.json"`

## Train AAE
2. `python -m scripts.train_aae -c "./settings/aae_hyperparams.json"`

## Input Optimization
3. `python -m scripts.optimize -c "settings/opt_hyperparams.json"`

## Resources
- <a href="https://github.com/fxia22/pointnet.pytorch" target="_blank">PointNet</a> implementation
- <a href="https://github.com/MaciejZamorski/3d-AAE" target="_blank">Adversarial Autoencoder</a> implementation