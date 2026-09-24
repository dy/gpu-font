"""Small fully convolutional classifier; mask batch padding, fold BN for deployment."""
import base64
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CHANNELS = [1, 16, 32, 48, 64]
STRIDES = [1, 2, 2, 2]
ARCH = 'font-conv16-32-48-64-v1'
CONTEXT_ARCH = 'font-conv16-32-48-64-64-v2'
CORPUS_ARCH = 'font-conv32-64-96-128-128-v3'
LARGE_ARCH = 'font-conv64-128-192-256-256-v4'
WIDER_ARCH = 'font-conv96-192-288-384-384-v5'
WIDEST_ARCH = 'font-conv128-256-384-512-512-v6'
ARCHITECTURES = {ARCH: CHANNELS, CONTEXT_ARCH: CHANNELS + [64], CORPUS_ARCH: [1, 32, 64, 96, 128, 128],
                 LARGE_ARCH: [1, 64, 128, 192, 256, 256], WIDER_ARCH: [1, 96, 192, 288, 384, 384],
                 WIDEST_ARCH: [1, 128, 256, 384, 512, 512]}


class Classifier(nn.Module):
    def __init__(self, classes=10, training=True, dilations=None, context=False, wide=False, large=False, architecture=None):
        super().__init__()
        if wide and not context: raise ValueError('Wide encoder requires context')
        if large and not wide: raise ValueError('Large encoder requires the wide architecture')
        if architecture and (context or wide or large): raise ValueError('Pass an architecture or its flags, not both')
        self.architecture = architecture or (LARGE_ARCH if large else CORPUS_ARCH if wide else CONTEXT_ARCH if context else ARCH)
        if self.architecture not in ARCHITECTURES: raise ValueError('Unknown architecture')
        self.context, self.wide, self.large = self.architecture != ARCH, self.architecture not in (ARCH, CONTEXT_ARCH), self.architecture not in (ARCH, CONTEXT_ARCH, CORPUS_ARCH)
        channels = ARCHITECTURES[self.architecture]
        self.strides = STRIDES + [1] if self.context else STRIDES
        self.dilations = dilations or [1] * len(self.strides)
        if len(self.dilations) != len(self.strides): raise ValueError('Wrong dilation count')
        self.convs = nn.ModuleList([nn.Conv2d(a, b, 3, stride=s, padding=d, dilation=d) for a, b, s, d in zip(channels, channels[1:], self.strides, self.dilations)])
        self.norms = nn.ModuleList([nn.BatchNorm2d(c) if training else nn.Identity() for c in channels[1:]])
        self.head = nn.Linear(channels[-1], classes)

    def forward(self, pixels, sizes=None):
        x = 1 - pixels
        if sizes is None:
            sizes = torch.tensor([pixels.shape[2:]], device=pixels.device).expand(len(x), 2)
        for conv, norm, stride in zip(self.convs, self.norms, self.strides):
            x = F.relu(norm(conv(x)))
            sizes = (sizes + stride - 1) // stride
            rows = torch.arange(x.shape[2], device=x.device)[None, :, None]
            cols = torch.arange(x.shape[3], device=x.device)[None, None, :]
            mask = (rows < sizes[:, 0, None, None]) & (cols < sizes[:, 1, None, None])
            x = x * mask[:, None]
        return self.head(x.sum((2, 3)) / sizes.prod(1)[:, None])

    def folded(self):
        result = Classifier(self.head.out_features, training=False, dilations=self.dilations, architecture=self.architecture)
        self.eval()
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            result.convs[i] = nn.utils.fuse_conv_bn_eval(conv, norm) if isinstance(norm, nn.BatchNorm2d) else copy.deepcopy(conv)
        result.head = copy.deepcopy(self.head)
        return result.eval()


def widen(model,noise=0,architecture=LARGE_ARCH):
    """Duplicate channels and divide incoming weights, preserving the initial function: any wider encoder of the same depth."""
    source,target=ARCHITECTURES[model.architecture],ARCHITECTURES.get(architecture)
    if not target or len(target)!=len(source) or target==source or any(t<s for s,t in zip(source,target)) or not np.isfinite(noise) or noise<0:raise ValueError('Expected a narrower encoder of the same depth and nonnegative noise')
    result=Classifier(model.head.out_features,training=isinstance(model.norms[0],nn.BatchNorm2d),dilations=model.dilations,architecture=architecture).to(next(model.parameters()).device)
    with torch.no_grad():
        for old,new,onorm,nnorm in zip(model.convs,result.convs,model.norms,result.norms):
            outs=torch.arange(new.out_channels,device=new.weight.device)%old.out_channels
            ins=torch.arange(new.in_channels,device=new.weight.device)%old.in_channels
            counts=torch.bincount(ins,minlength=old.in_channels)
            new.weight.copy_(old.weight[outs][:,ins]/counts[ins][None,:,None,None]);new.bias.copy_(old.bias[outs])
            if noise:new.weight[old.out_channels:].add_(torch.randn_like(new.weight[old.out_channels:])*noise)
            if isinstance(onorm,nn.BatchNorm2d):
                for key,value in onorm.state_dict().items():nnorm.state_dict()[key].copy_(value if value.ndim==0 else value[outs])
        ins=torch.arange(result.head.in_features,device=result.head.weight.device)%model.head.in_features;counts=torch.bincount(ins,minlength=model.head.in_features)
        result.head.weight.copy_(model.head.weight[:,ins]/counts[ins][None,:]);result.head.bias.copy_(model.head.bias)
    return result.train(model.training)


def pack_bits(values, bits):
    """Signed integers `bits` apiece in a little-endian bit stream, value i filling bits i * bits onward, as
    src/catalog.mjs pack() writes them. At 8 bits these are plain int8 bytes."""
    v = np.asarray(values, np.int64).ravel() & ((1 << bits) - 1)
    return np.packbits(((v[:, None] >> np.arange(bits)) & 1).astype(np.uint8).ravel(), bitorder='little').tobytes()


def unpack_bits(data, bits, count):
    if len(data) != -(-count * bits // 8): raise ValueError('Wrong packed length')
    v = (np.unpackbits(np.frombuffer(data, np.uint8), bitorder='little')[:count * bits].reshape(count, bits).astype(np.int64) << np.arange(bits)).sum(1)
    return np.where(v >= 1 << (bits - 1), v - (1 << bits), v).astype(np.int8)


def load_export(artifact):
    if artifact['version'] != 1 or artifact['architecture'] not in ARCHITECTURES:
        raise ValueError('Unsupported classifier')
    model = Classifier(len(artifact['fonts']), training=False, dilations=artifact.get('dilations'), architecture=artifact['architecture']).eval()
    modules = [*model.convs, model.head]
    if len(artifact['layers']) != len(modules):
        raise ValueError('Wrong layer count')
    with torch.no_grad():
        for layer, target in zip(artifact['layers'], modules):
            if layer['shape'] != list(target.weight.shape):
                raise ValueError('Wrong layer shape')
            values = unpack_bits(base64.b64decode(layer['weights'], validate=True), layer.get('bits', 8), target.weight.numel()).reshape(len(target.weight), -1)
            weights = values.astype(np.float32) * np.array(layer['scale'], dtype=np.float32)[:, None]
            target.weight.copy_(torch.from_numpy(weights.reshape(layer['shape'])))
            target.bias.copy_(torch.tensor(layer['bias']))
    return model


def export(model, fonts, preparation, bits=8):
    """Weights rounded per output row to `bits` (8, or 6 for the shipped style encoder: bench/style.md, Quantization)."""
    model = model.folded(); levels = 2 ** (bits - 1) - 1
    layers = []
    restored = Classifier(len(fonts), training=False, dilations=model.dilations, architecture=model.architecture).eval()
    for source, target in zip([*model.convs, model.head], [*restored.convs, restored.head]):
        weight = source.weight.detach().numpy()
        row = weight.reshape(len(weight), -1)
        scale = np.maximum(np.max(np.abs(row), axis=1) / levels, 1e-12).astype(np.float32)
        quant = np.clip(np.round(row / scale[:, None]), -levels, levels).astype(np.int8)
        decoded = quant.astype(np.float32) * scale[:, None]
        with torch.no_grad():
            target.weight.copy_(torch.from_numpy(decoded.reshape(weight.shape)))
            target.bias.copy_(source.bias)
        layers.append({'shape': list(weight.shape), 'scale': scale.tolist(), **({'bits': bits} if bits != 8 else {}),
                       'weights': base64.b64encode(pack_bits(quant, bits)).decode(), 'bias': source.bias.detach().tolist()})
    return {'version': 1, 'architecture': model.architecture, 'dilations': model.dilations, 'fonts': fonts, 'preparation': preparation, 'layers': layers}, restored
