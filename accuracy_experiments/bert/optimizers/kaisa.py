import torch
from .kaisa_utils.preconditioner import KFACPreconditioner
import sys
try:
    from apex.optimizers import FusedLAMB
except:
    from .lamb import LAMBOptimizer as FusedLAMB


class KAISAOptimizer(FusedLAMB):
    def __init__(self, model, lr=1e-3, momentum=0.9, weight_decay=0, factor_decay=0.95, damping=1e-3, kl_clip=1e-3, TInv=100, TCov=10, measure_time=False):
        super(KAISAOptimizer, self).__init__(model.parameters(), lr=lr, weight_decay=weight_decay)
        # self.model = model
        self.preconditioner = KFACPreconditioner(model=model, skip_layers=["Linear(in_features=1024, out_features=30528, bias=False)"], lr=lambda x: self.param_groups[0]['lr'], factor_decay=factor_decay, damping=damping, kl_clip=kl_clip, inv_update_steps=TInv, factor_update_steps=TCov, update_factors_in_hook=False, measure_time=measure_time)
        self.timer = self.preconditioner.timer
    def step(self, closure=None):
        self.preconditioner.step()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0, norm_type='inf')
        super(KAISAOptimizer, self).step(closure=closure)
        