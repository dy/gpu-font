"""Independent CPU PyTorch outputs for a candidate artifact, without demo changes."""
import sys
from pathlib import Path
import numpy as np
import torch
from train.robustness import read, write, sha
from train.ten_model import load_export

path=Path(sys.argv[1]);out=Path(sys.argv[2]);torch.set_num_threads(2)
model=load_export(read(path));cases=[]
for width,height in [(1,1),(128,48),(127,47),(7,3)]:
    pixels=(np.arange(width*height)%113/112).astype(np.float32)
    with torch.no_grad():logits=model(torch.from_numpy(pixels.reshape(1,1,height,width)))[0].tolist()
    cases.append({'width':width,'height':height,'pixels':pixels.tolist(),'logits':logits})
write(out,{'modelSha256':sha(path),'cases':cases})
