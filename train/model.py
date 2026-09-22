"""Small offline encoder. Exported weights have no PyTorch runtime dependency."""
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = 'conv12-24-32-48-gap-32-v1'


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = [1, 12, 24, 32, 48]
        for a, b in zip(channels, channels[1:]):
            layers.extend([nn.Conv2d(a, b, 3, stride=2, padding=1), nn.ReLU()])
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.project = nn.Linear(48, 32)

    def forward(self, pixels):
        return F.normalize(self.project(self.features(1 - pixels)), dim=1)


def contrastive_loss(vectors, labels, temperature=.15):
    """Each anchor uses other samples of its family as positives; excludes itself."""
    n = len(vectors)
    if n < 2:
        raise ValueError('Every anchor needs a second sample of its family')
    eye = torch.eye(n, dtype=torch.bool, device=vectors.device)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    if not positives.any(dim=1).all():
        raise ValueError('Every anchor needs a second sample of its family')
    logits = vectors @ vectors.T / temperature
    logits = logits.masked_fill(eye, -1e9)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_probs.masked_fill(~positives, 0).sum(dim=1) / positives.sum(dim=1)).mean()


def export_model(model):
    return {'architecture': ARCHITECTURE, 'tensors': {
        name: {'shape': list(value.shape), 'values': value.detach().cpu().flatten().tolist()}
        for name, value in model.state_dict().items()
    }}


def load_export(artifact):
    if artifact['architecture'] != ARCHITECTURE:
        raise ValueError('Unsupported architecture')
    model = Encoder()
    expected = model.state_dict()
    tensors = artifact['tensors']
    if set(expected) != set(tensors):
        raise ValueError('Unexpected tensor names')
    state = {}
    for name, value in expected.items():
        tensor = tensors[name]
        if tensor['shape'] != list(value.shape) or len(tensor['values']) != value.numel():
            raise ValueError(f'Tensor shape mismatch: {name}')
        values = torch.tensor(tensor['values'], dtype=torch.float32)
        if not torch.isfinite(values).all():
            raise ValueError(f'Nonfinite weights: {name}')
        state[name] = values.reshape(value.shape)
    model.load_state_dict(state)
    return model.eval()
