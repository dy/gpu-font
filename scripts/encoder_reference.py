"""Independent PyTorch projection vectors for the shared encoder's WebGPU check."""
import sys
from pathlib import Path

import numpy as np
import torch

from train.encoder import load_encoder,unit
from train.robustness import sha
from train.encoder_data import save


def main(path,destination):
    torch.set_num_threads(2);model=load_encoder(path).eval();rng=np.random.default_rng(416);cases=[]
    with torch.no_grad():
        for height,width in [(1,1),(48,128),(47,127),(3,7)]:
            pixels=rng.random((height,width),dtype=np.float32)
            vector=model(torch.from_numpy(pixels)[None,None]).numpy()
            cases.append({'width':width,'height':height,'pixels':pixels.flatten().tolist(),'projection':vector[0].tolist(),'embedding':unit(vector)[0].tolist()})
    save(destination,{'encoderSha256':sha(path),'cases':cases})


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
