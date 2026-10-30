"""Preserve the original BF16 residual arithmetic through the shared hook API."""
from repsteer.interventions import Intervention


def unit(v):
    return v/(v.norm()+1e-8)


class NumericIntervention(Intervention):
    def transform(self,residual):
        if residual.shape[-1]!=self.direction.numel():
            raise ValueError('Direction width differs from residual width')
        v=unit(self.direction.to(device=residual.device,dtype=residual.dtype))
        if self.mode=='add':
            if self.scale!='absolute':
                raise ValueError('These frozen recipes use absolute add strength')
            return residual+float(self.strength)*v
        component=(residual@v).unsqueeze(-1)
        if self.mode=='ablate':
            return residual-float(self.strength)*(component*v)
        target=unit(self.target.to(device=residual.device,dtype=residual.dtype))
        return residual-float(self.strength)*component*v+float(self.strength)*component*target
