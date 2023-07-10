import os
import torch

f = 'runs/train/exp_v7_CoT_CoordAtt/weights/best_292.pt'
# Strip optimizer from 'f' to finalize training, optionally save as 's'
x = torch.load(f, map_location=torch.device('cpu'))
if x.get('ema'):
    x['model'] = x['ema']  # replace model with ema
for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
    x[k] = None
x['epoch'] = -1
x['model'].half()  # to FP16
for p in x['model'].parameters():
    p.requires_grad = False
torch.save(x, f)
mb = os.path.getsize(f) / 1E6  # filesize
print(f"Optimizer stripped from {f},{(' saved as %s,' % f) if f else ''} {mb:.1f}MB")