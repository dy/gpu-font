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


class Classifier(nn.Module):
    def __init__(self, classes=10, training=True, dilations=None, context=False):
        super().__init__()
        self.context = context
        self.architecture = CONTEXT_ARCH if context else ARCH
        channels = CHANNELS + [64] if context else CHANNELS
        self.strides = STRIDES + [1] if context else STRIDES
        self.dilations = dilations or [1] * len(self.strides)
        if len(self.dilations) != len(self.strides): raise ValueError('Wrong dilation count')
        self.convs = nn.ModuleList([nn.Conv2d(a, b, 3, stride=s, padding=d, dilation=d) for a, b, s, d in zip(channels, channels[1:], self.strides, self.dilations)])
        self.norms = nn.ModuleList([nn.BatchNorm2d(c) if training else nn.Identity() for c in channels[1:]])
        self.head = nn.Linear(CHANNELS[-1], classes)

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
        result = Classifier(self.head.out_features, training=False, dilations=self.dilations, context=self.context)
        self.eval()
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            result.convs[i] = nn.utils.fuse_conv_bn_eval(conv, norm) if isinstance(norm, nn.BatchNorm2d) else copy.deepcopy(conv)
        result.head = copy.deepcopy(self.head)
        return result.eval()


def load_export(artifact):
    if artifact['version'] != 1 or artifact['architecture'] not in [ARCH, CONTEXT_ARCH]:
        raise ValueError('Unsupported classifier')
    model = Classifier(len(artifact['fonts']), training=False, dilations=artifact.get('dilations'), context=artifact['architecture'] == CONTEXT_ARCH).eval()
    modules = [*model.convs, model.head]
    if len(artifact['layers']) != len(modules):
        raise ValueError('Wrong layer count')
    with torch.no_grad():
        for layer, target in zip(artifact['layers'], modules):
            if layer['shape'] != list(target.weight.shape):
                raise ValueError('Wrong layer shape')
            values = np.frombuffer(base64.b64decode(layer['weights'], validate=True), dtype=np.int8).reshape(len(target.weight), -1)
            weights = values.astype(np.float32) * np.array(layer['scale'], dtype=np.float32)[:, None]
            target.weight.copy_(torch.from_numpy(weights.reshape(layer['shape'])))
            target.bias.copy_(torch.tensor(layer['bias']))
    return model


def export(model, fonts, preparation):
    model = model.folded()
    layers = []
    restored = Classifier(len(fonts), training=False, dilations=model.dilations, context=model.context).eval()
    for source, target in zip([*model.convs, model.head], [*restored.convs, restored.head]):
        weight = source.weight.detach().numpy()
        row = weight.reshape(len(weight), -1)
        scale = np.maximum(np.max(np.abs(row), axis=1) / 127, 1e-12).astype(np.float32)
        quant = np.clip(np.round(row / scale[:, None]), -127, 127).astype(np.int8)
        decoded = quant.astype(np.float32) * scale[:, None]
        with torch.no_grad():
            target.weight.copy_(torch.from_numpy(decoded.reshape(weight.shape)))
            target.bias.copy_(source.bias)
        layers.append({'shape': list(weight.shape), 'scale': scale.tolist(),
                       'weights': base64.b64encode(quant.tobytes()).decode(), 'bias': source.bias.detach().tolist()})
    return {'version': 1, 'architecture': model.architecture, 'dilations': model.dilations, 'fonts': fonts, 'preparation': preparation, 'layers': layers}, restored
