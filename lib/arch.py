import torch
import torch.nn as nn


def get_backbone(model):
    model_type = model.base_model_prefix
    backbone = getattr(model, model_type)
    return backbone


def get_encoder(model):
    backbone = get_backbone(model)
    encoder = backbone.decoder if ('opt' in model.config.model_type) else backbone.encoder
    return encoder


def get_layers(model):
    if 'opt' in model.config.model_type:
        encoder = get_encoder(model)
        layers = encoder.layers 
    else:
        layers = model.model.layers
    return layers


def get_mha(model, index):
    layer = get_layers(model)[index]
    mha_proj = layer.self_attn
    return mha_proj


def get_mha_proj(model, index):
    layer = get_layers(model)[index]
    mha_proj = layer.self_attn.out_proj if ('opt' in model.config.model_type) else layer.self_attn.o_proj
    return mha_proj


def get_ffn1(model, index):
    layer = get_layers(model)[index]
    ffn1 = layer.fc1 if ('opt' in model.config.model_type) else layer.intermediate
    return ffn1

def get_ffn2(model, index):
    layer = get_layers(model)[index]
    ffn2 = layer.fc2 if ('opt' in model.config.model_type) else layer.output
    return ffn2

def get_gate(model, index):
    layer = get_layers(model)[index]
    # 检查模型类名，动态选择路径
    if "Qwen" in model.__class__.__name__:
        proj = layer.mlp.gate_proj # Qwen的路径
    else:
        proj = layer.mlp.gate_proj # Llama的路径 - 修正：也在mlp子模块中
    return proj

def get_up(model, index):
    layer = get_layers(model)[index]
    # 检查模型类名，动态选择路径
    if "Qwen" in model.__class__.__name__:
        proj = layer.mlp.up_proj # Qwen的路径
    else:
        proj = layer.mlp.up_proj # Llama的路径 - 修正：也在mlp子模块中
    return proj

def get_down(model, index):
    layer = get_layers(model)[index]
    # 检查模型类名，动态选择路径
    if "Qwen" in model.__class__.__name__:
        proj = layer.mlp.down_proj # Qwen的路径
    else:
        proj = layer.mlp.down_proj # Llama的路径 - 修正：也在mlp子模块中
    return proj

def get_mha_ln(model, index):
    layer = get_layers(model)[index]
    ln = layer.self_attn_layer_norm if ('opt' in model.config.model_type) else layer.LayerNorm
    return ln


def get_ffn_ln(model, index):
    layer = get_layers(model)[index]
    ln = layer.final_layer_norm if ('opt' in model.config.model_type) else layer.LayerNorm
    return ln


def get_norm_layer(module, layers=[nn.LayerNorm]):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return module
    res = []
    for name1, child in module.named_children():
        res.append(get_norm_layer(child, layers=layers))
    return res


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def get_classifier(model):
    backbone = get_backbone(model)
    if backbone.pooler is not None:
        classifier = model.classifier
    else:
        classifier = model.classifier.out_proj
    return classifier
