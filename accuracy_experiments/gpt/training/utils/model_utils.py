from models.cifar import (alexnet, densenet, resnet,
                          vgg16_bn, vgg19_bn,
                          wrn)
from models.lra import lra_model
from models.t2t_vit.t2t_vit import t2t_vit_t_14, t2t_vit_t_24
from models.vit.vit import vit_base_patch16, vit_large_patch16, vit_huge_patch14
from timm.models.levit import Levit
from models.bert.bert import *
from models.linear.linear import linear
from models.linear.autoencoder import autoencoder
import torchvision.models as models
from models.mae.mae import mae_vit_base_patch16_dec512d8b, mae_vit_large_patch16_dec512d8b, mae_vit_huge_patch14_dec512d8b


def get_model(network, **kwargs):
    networks = {
        'alexnet': alexnet,
        'densenet': densenet,
        'resnet': resnet,
        'vgg16_bn': vgg16_bn,
        'vgg19_bn': vgg19_bn,
        'wrn': wrn,
        'lra': lra_model,
        't2t_vit_14': t2t_vit_t_14,
        't2t_vit_24': t2t_vit_t_24,
        'vit_base_patch16': vit_base_patch16,
        'vit_large_patch16': vit_large_patch16,
        'vit_huge_patch14': vit_huge_patch14,
        'mae_base': mae_vit_base_patch16_dec512d8b,
        'mae_large': mae_vit_large_patch16_dec512d8b,
        'mae_huge': mae_vit_huge_patch14_dec512d8b,
        'bert_large_cased': bert_large_cased,
        'bert_base_cased': bert_base_cased,
        'bert_large_uncased': bert_large_uncased,
        'bert_base_uncased': bert_base_uncased,
        'bert_base_cased_mlm': bert_base_cased_mlm,
        'bert_large_cased_mlm': bert_large_cased_mlm,
        'bert_base_uncased_mlm': bert_base_uncased_mlm,
        'bert_large_uncased_mlm': bert_large_uncased_mlm,
        'bert_large_uncased_nvidia': bert_large_uncased_nvidia,
        'bert_base_cased_classification': bert_base_cased_sequence_classification,
        'bert_large_cased_classification': bert_large_cased_sequence_classification,
        'bert_large_uncased_classification_nvidia': bert_large_uncased_classification_nvidia,
        'bert_large_cased_question_answering': bert_large_cased_question_answering,
        'bert_large_uncased_question_answering_nvidia': bert_large_uncased_question_answering_nvidia,
        'bert_base_cased_question_answering': bert_base_cased_question_answering,
        'transformer': transformer,
        'linear': linear,
        'levit': Levit,
        'autoencoder': autoencoder,
    }
    if network == "levit":
        kwargs = {'img_size': kwargs['img_size'], 'num_classes': kwargs['num_classes']}
    if network.startswith('resnet') and kwargs['num_classes'] == 1000:
        return get_imagenet_resnet(network + str(kwargs['depth']))
    else:
        return networks[network](**kwargs)


def get_imagenet_resnet(network):
    if network == 'resnet18':
        return models.resnet18(pretrained=False)
    elif network == 'resnet34':
        return models.resnet34(pretrained=False)
    elif network == 'resnet50':
        return models.resnet50(pretrained=False)
    elif network == 'resnet101':
        return models.resnet101(pretrained=False)
    elif network == 'resnet152':
        return models.resnet152(pretrained=False)
    else:
        raise NotImplementedError
