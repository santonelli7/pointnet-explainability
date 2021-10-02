import json
import argparse
from os.path import join
from tqdm import tqdm

from models.pointnet import PointNet
from models.aae import AAE

import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl

def models_sanity_check(random_noise_input, gt_class, pointnet, encoder, generator):
    assert not gt_class.requires_grad, "Ground truth does not require gradients"
    assert random_noise_input.requires_grad, "Input does require gradients"

    assert (not pointnet.training and not encoder.training and not generator.training), "The models need to be in evaluation mode"

    # PointNet must be freezed
    for param in pointnet.parameters():
        assert not param.requires_grad, "The comparator model must be frozen"

    # Encoder must be freezed
    for param in encoder.parameters():
        assert not param.requires_grad, "The encoder model must be frozen"

    # Generator must be freezed
    for param in generator.parameters():
        assert not param.requires_grad, "The generator model must be frozen"
    
def get_class_distribution(classes, log_probs):
    values = torch.exp(log_probs.squeeze()).tolist()
    data = [[label, val] for (label, val) in zip(classes, values)]
    table = wandb.Table(data=data, columns = ["class", "probability"])
    return table

def optimization_loop(config, pointnet, encoder, generator, kl_div, optimizer, points, ground_truth):
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset as ShapeNet
        classes = list(ShapeNet.synth_id_to_category.values())
    elif dataset_name == 'modelnet':
        from datasets.modelnet import ModelNet40 as ModelNet
        classes = ModelNet.all_classes
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or 'f'`modelnet`. Got: `{dataset_name}`')
    
    # Optimization loop
    encoder.eval()    
    generator.eval()
    pointnet.eval()
    models_sanity_check(points, ground_truth, pointnet, encoder, generator)

    progress_bar = tqdm(range(1), desc='Optimizing')
    it = 0
    convergence = 1
    while convergence > config["threshold"]:
        optimizer.zero_grad()
        
        code, _, _ = encoder(points)
        gen_points = generator(code)
        log_probs, _, _ = pointnet(gen_points)

        assert log_probs.shape == ground_truth.shape
        loss = kl_div(log_probs, ground_truth)
        loss.backward()
        optimizer.step()
        convergence = loss.item()
        progress_bar.set_postfix({'loss': convergence, 'iteration': it})

        wandb.log({'loss': convergence,
                   'iteration': it,
                }),
        wandb.log({'pointcloud': wandb.Object3D(points.detach().cpu().squeeze().transpose(0,1).numpy())})
        wandb.log({'gen_pointcloud': wandb.Object3D(gen_points.detach().cpu().squeeze().transpose(0,1).numpy())})
        # if it % 1000 == 0:
        table = get_class_distribution(classes, log_probs.detach())
        wandb.log({"class_distribution" : wandb.plot.bar(table, "class", "probability", title="Class Distribution")})

        it += 1

    # table = get_class_distribution(classes, log_probs.detach())
    # wandb.log({"class_distribution" : wandb.plot.bar(table, "class", "probability", title="Class Distribution")})

def main(config):
    pl.seed_everything(config['seed'])

    # MODELS
    pointnet = PointNet.load_from_checkpoint(join(config['ckpts_dir'], config['pointnet_ckpt']), map_location=config["device"])
    pointnet = pointnet.to(config["device"])
    pointnet.freeze()

    with open(join("settings", config["aae_config"])) as f:
        aae_config = json.load(f)
    aae = AAE.load_from_checkpoint(join(config['ckpts_dir'], config['aae_ckpt']), config=aae_config, map_location=config["device"])
    aae = aae.to(config["device"])
    aae.freeze()
    encoder = aae.encoder
    generator = aae.generator

    # LOSS
    kl_div = nn.KLDivLoss(reduction='batchmean')

    # INPUT
    random_noise_points = torch.randn((1, 3, config["num_points"]), device=config["device"], requires_grad=True)

    # GROUND TRUTH
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset as ShapeNet
        cls_idx = ShapeNet.category_to_synth_id[ShapeNet.synth_id_to_number[config['expected_class']]]
        classes_idx = list(ShapeNet.synth_id_to_number.values())
    elif dataset_name == 'modelnet':
        from datasets.modelnet import ModelNet40 as ModelNet
        cls_idx = ModelNet.category_to_number[config['expected_class']]
        classes_idx = list(ModelNet.category_to_number.values())
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or `modelnet`. Got: `{dataset_name}`')

    gt_class = (torch.Tensor(classes_idx) == cls_idx).unsqueeze(dim=0).float().to(config["device"])

    # OPTIMIZER
    optim = getattr(torch.optim, config['optimizer']['type'])
    optim = optim([random_noise_points], 
        **config['optimizer']['hyperparams'])

    wandb.init(id=config["run_id"], project=config["project_name"], config=config, resume="allow")

    optimization_loop(config, pointnet, encoder, generator, kl_div, optim, random_noise_points, gt_class)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path')
    args = parser.parse_args()

    config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
    assert config is not None

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    main(config)
