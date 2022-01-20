import json
import argparse
from os.path import join
from tqdm import tqdm

from models.aae import AAE

import wandb
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

def models_sanity_check(code, ground_truth, pointnet, generator):
    assert not ground_truth.requires_grad, "Ground truth does not require gradients"
    assert code.requires_grad, "Input does require gradients"

    assert (not pointnet.training and not generator.training), "The models need to be in evaluation mode"

    # PointNet must be freezed
    for param in pointnet.parameters():
        assert not param.requires_grad, "The comparator model must be frozen"

    # Generator must be freezed
    for param in generator.parameters():
        assert not param.requires_grad, "The generator model must be frozen"
    
def get_class_distribution(config, logits):
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset as ShapeNet
        classes = list(ShapeNet.synth_id_to_category.values())
    elif dataset_name == 'modelnet':
        from datasets.modelnet import ModelNet40 as ModelNet
        classes = ModelNet.all_classes
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or 'f'`modelnet`. Got: `{dataset_name}`')

    probs = F.softmax(logits, dim=1)
    values = probs.squeeze().tolist()
    data = [[label, val] for (label, val) in zip(classes, values)]
    table = wandb.Table(data=data, columns=["class", "probability"])
    return table, values

def optimization_loop(config, pointnet, generator, code, ground_truth, cls_idx):

    code.requires_grad_()
    generator.eval()
    pointnet.eval()
    models_sanity_check(code=code, ground_truth=ground_truth, pointnet=pointnet, generator=generator)

    # Optimization loop
    progress_bar = tqdm(range(1), desc='Optimizing', bar_format='{desc} [{elapsed}, {postfix}]')
    it = 0
    while it < config['max_iters']:
        
        gen_points = generator(code)
        logits = pointnet(gen_points)
        activation = logits[0, cls_idx]

        # L1 regularizer to minimize to inject sparsity
        loss = activation - config['w_decay'] * torch.norm(gen_points, p=1)
        loss.backward()
    
        # Gradient ascent
        grad = code.grad.data

        # Normalize the gradients (make them have mean = 0 and std = 1)
        g_std = torch.std(grad)
        g_mean = torch.mean(grad)
        smooth_grad = (grad - g_mean) / g_std

        code.data += config['lr'] * smooth_grad

        code.grad.data.zero_()
        
        # Logs
        if it % 100 == 0:
            table, probs = get_class_distribution(config, logits.detach())
            wandb.log({'gen_pointcloud': wandb.Object3D(gen_points.detach().cpu().squeeze().numpy().transpose())})
            wandb.log({"class_distribution" : wandb.plot.bar(table, "class", "probability", title="Class Distribution")})

        # table, probs = get_class_distribution(config, logits.detach())
        # wandb.log({'gen_pointcloud': wandb.Object3D(gen_points.detach().cpu().squeeze().numpy().transpose())})
        # wandb.log({"class_distribution" : wandb.plot.bar(table, "class", "probability", title="Class Distribution")})

        prob = F.softmax(logits.detach(), dim=1)[0, cls_idx].item()

        progress_bar.set_postfix({'loss': loss.item(),
                                  'activation': activation.item(), 
                                  'prob': prob,
                                  'iteration': it,
                                  })
        wandb.log({'loss': loss.item(),
                   'activation': activation.item(), 
                   'iteration': it,
                })
            
        it += 1

    table, probs = get_class_distribution(config, logits.detach())
    wandb.log({'gen_pointcloud': wandb.Object3D(gen_points.detach().cpu().squeeze().numpy().transpose())})
    wandb.log({"class_distribution" : wandb.plot.bar(table, "class", "probability", title="Class Distribution")})

def main(config):
    pl.seed_everything(config['seed'])

    # MODELS
    with open(join("settings", config["aae_config"])) as f:
        aae_config = json.load(f)
    aae = AAE.load_from_checkpoint(join(config['ckpts_dir'], config['aae_ckpt']), config=aae_config, map_location=config["device"])
    aae.freeze()
    generator = aae.generator.to(config["device"])
    comparator = aae.comparator.to(config["device"])

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

    # INPUT RANDOM NOISE
    code = torch.FloatTensor(1, config["z_size"])
    code = code.to(config["device"])
    code.uniform_(0, 1)

    wandb.init(id=config["run_id"], project=config["project_name"], config=config, resume="allow")

    wandb.run.name = f"{config['expected_class']}_vis"

    optimization_loop(config=config, pointnet=comparator, generator=generator, code=code, ground_truth=gt_class, cls_idx=cls_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path')
    parser.add_argument('-C', '--cls', default='airplane', type=str, help='expected class')
    args = parser.parse_args()
    expected_class = args.cls

    config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
    assert config is not None

    config['expected_class'] = expected_class

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    main(config)
