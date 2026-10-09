"""Load exact official neural-network definitions without Dassl dependency.

AST selects unmodified class definitions, not a reimplementation. Excluded
parts are dataset/trainer registration and the historical visualization code.
Source SHA256 and upstream commit are recorded for auditability.
"""
import ast
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from clip import clip
from clip.model import VisionTransformer
from clip.simple_tokenizer import SimpleTokenizer


def load_official(kind):
    name = 'clip_vpn_cnn_image.py' if kind == 'image' else 'clip_vpn.py'
    source = Path(__file__).resolve().parents[1] / 'trainers' / name
    text = source.read_text()
    tree = ast.parse(text)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name not in ['CLIP_VPN', 'CLIP_VPN_cnn_image']:
            assert not node.decorator_list
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CUSTOM_TEMPLATES' for t in node.targets):
            nodes.append(node)
    env = dict(torch=torch, nn=nn, F=F, math=math, sqrt=math.sqrt,
               normalize=F.normalize, clip=clip, VisionTransformer=VisionTransformer,
               _tokenizer=SimpleTokenizer())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), env)
    return SimpleNamespace(**env), dict(source=str(source), sha256=hashlib.sha256(text.encode()).hexdigest(),
                                      upstream='c6f89a4bcb76a33fc8ad3a7e7992b5c110630c00', definitions='unmodified AST classes')
