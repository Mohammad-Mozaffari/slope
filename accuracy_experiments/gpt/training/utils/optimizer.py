from optimizers import (KFACOptimizer, EKFACOptimizer, HyLoOptimizer, MKOROptimizer, KAISAOptimizer, HKOROptimizer, LAMBOptimizer)
import torch


def set_no_decay(model, model_name, optimizer):
    if "bert" in model_name or "mae" in model_name:
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
        no_decay_param_group = {"params": [], "weight_decay": 0.0}
        for key, value in optimizer.param_groups[0].items():
            if key not in ["params", "weight_decay"]:
                no_decay_param_group[key] = value
        params = optimizer.param_groups[0]["params"]
        for name, param in model.named_parameters():
            if any(nd in name for nd in no_decay):
                no_decay_param_group["params"].append(param)
                for i in range(len(params)):
                    if param.shape == params[i].shape and torch.all(param == params[i]):
                        del params[i]
                        break
        optimizer.param_groups.append(no_decay_param_group)
        print("Disabled weight decay for ", no_decay)
    return optimizer


def get_optimizer(args, model, backend, momentum=0.9, stat_decay=0.95, damping=1e-3, kl_clip=1e-2, TCov=10, TScal=10,
                  iters=None, warmup_epochs=5, vocab_size=None, grad_scale=1.0):
    name = args.optimizer
    model_name = args.model_name
    lr = args.lr
    weight_decay = args.weight_decay
    measure_time = args.time
    inv_freq = args.inv_freq
    grad_accum_steps = args.grad_accum_iter

    TCov = min(TCov, inv_freq)
    TScal = min(TScal, inv_freq)
    if name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'kfac':
        optimizer = KFACOptimizer(model, lr, momentum=momentum, stat_decay=stat_decay, damping=damping,
                                  kl_clip=kl_clip, weight_decay=weight_decay, TCov=TCov, TInv=inv_freq,
                                  measure_time=measure_time, backend=backend)
    elif name == 'ekfac':
        optimizer = EKFACOptimizer(model, lr, momentum=momentum, stat_decay=stat_decay, damping=damping,
                                   kl_clip=kl_clip, weight_decay=weight_decay, TCov=TCov, TScal=TScal, TInv=inv_freq)
    elif name.startswith('hylo'):
        kis_only = name.endswith('kis')
        optimizer = HyLoOptimizer(model, lr, weight_decay=weight_decay, momentum=momentum, freq=inv_freq,
                                  kl_clip=kl_clip, iters=iters, warmup_epochs=warmup_epochs, measure_time=measure_time,
                                  kis_only=kis_only, backend=backend)
    elif name.startswith('mkor'):
        sgd_layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear) and \
                      vocab_size is not None and module.out_features == vocab_size]
        svd = name.endswith('svd')
        optimizer = MKOROptimizer(model, lr, momentum=momentum, stat_decay=stat_decay, damping=damping, kl_clip=kl_clip,
                                  weight_decay=weight_decay, inv_freq=inv_freq, measure_time=measure_time, svd=svd,
                                  backend=backend, grad_accum_steps=grad_accum_steps, sgd_layers=sgd_layers,
                                  grad_scale=grad_scale)
    elif name.startswith('hkor'):
        optimizer = HKOROptimizer(model, lr, momentum=momentum, stat_decay=stat_decay, damping=damping, kl_clip=kl_clip,
                             weight_decay=weight_decay, inv_freq=inv_freq, measure_time=measure_time, svd=False,
                             backend=backend)
    elif name == "kaisa":
        optimizer = KAISAOptimizer(model, lr, momentum=momentum, weight_decay=weight_decay, factor_decay=stat_decay,
                                   damping=damping, kl_clip=kl_clip, TInv=inv_freq, TCov=TCov,
                                   measure_time=measure_time)
    elif name == "lamb":
        optimizer = LAMBOptimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Invalid optimizer name")

    optimizer = set_no_decay(model, model_name, optimizer)

    return optimizer
