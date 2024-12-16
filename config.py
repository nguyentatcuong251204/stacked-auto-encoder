import torch
from torch import nn
from model.ae import Encoder as AEncoder
from model.ae import Decoder as ADecoder
from model.vae import Encoder as VEncoder
from model.vae import Decoder as VDecoder

from model.ae_pretrain import Encoder as PreTrain_AEncoder
from model.ae_pretrain import Decoder as PreTrain_ADecoder
from model.vae_pretrain import Encoder as PreTrain_VEncoder
from model.vae_pretrain import Decoder as PreTrain_VDecoder



pre2_checkpoint = [r"D:\Paper\checkpoint_resnet_no_noise\1sae\encoder1.pth"]

pre3_checkpoint = [r"D:\Paper\checkpoint_resnet__no_noise\2sae\encoder1.pth",
                    r"D:\Paper\checkpoint_resnet_no_noise\2sae\encoder2.pth"]

pre4_checkpoint = [r"D:\Paper\checkpoint_resnet_no_noise\3sae\encoder1.pth",
                    r"D:\Paper\checkpoint_resnet_no_noise\3sae\encoder2.pth",
                    r"D:\Paper\checkpoint_resnet_no_noise\3sae\encoder3.pth"]
experiment_config = {
    "clean_folder_path": r"D:\Paper\train\HR",
    "soil_folder_path": r"D:\Paper\train\LR",
    "batch_size": 16,
    "lr": 1e-4,
    "device": "cuda",
    "epoch": 100,
    "Model": {
        "AE": {
            # "is_stack" : False,
            # "previous_encoder": None,
            "encoder": PreTrain_AEncoder(n_block=4),
            "decoder": PreTrain_ADecoder(n_block=4),
            "is_variant": False,
            "previous_checkpoint": None,
            "checkpoint": [r"D:\Paper\checkpoint_resnet_no_noise\ae\encoder1.pth",
                           r"D:\Paper\checkpoint_resnet_no_noise\ae\encoder2.pth",
                           r"D:\Paper\checkpoint_resnet_no_noise\ae\encoder3.pth",
                           r"D:\Paper\checkpoint_resnet_no_noise\ae\encoder4.pth",
                           r"D:\Paper\checkpoint_resnet_no_noise\ae\decoder.pth"]
        },
        "VAE": {
            # "is_stack" : False,
            # "previous_encoder": None,
            "encoder": PreTrain_VEncoder(),
            "decoder": PreTrain_VDecoder(),
            "is_variant": True,
            "previous_checkpoint": None,
            "checkpoint": [r"D:\Paper\checkpoint_resnet_no_noise\vae\encoder.pth",
                           r"D:\Paper\checkpoint_resnet_no_noise\vae\decoder.pth"]
                           
        },
        "1SAE": {
            # "is_stack" : False,
            # "previous_encoder": None,
            "encoder": PreTrain_AEncoder(1),
            "decoder": PreTrain_ADecoder(1),
            "is_variant": False,
            "previous_checkpoint": None,
            "checkpoint": [r"D:\Paper\checkpoint_resnet_no_noise\1sae\encoder1.pth",
                           r"D:\Paper\checkpoint_resnet_no_noise\1sae\decoder1.pth"]
        },
        "2SAE": {
            # "is_stack": True,
            # "previous_encoder": [AEncoder(3,[8])],
            "encoder": PreTrain_AEncoder(2, pre2_checkpoint),
            "decoder": PreTrain_ADecoder(2),
            "is_variant": False,
            "previous_checkpoint": [r"D:\Paper\checkpoint_resnet_no_noise\1sae\encoder1.pth"],
            "checkpoint": [r"D:\Paper\checkpoint_resnet_no_noise\2sae\encoder1.pth",
                           r"D:\Paper\checkpoint_resnet_no_noise\2sae\encoder2.pth",
                           r"D:\Paper\checkpoint_resnet_no_noise\2sae\decoder2.pth"]
        },
        "3SAE": {
            # "is_stack": True,
            # "previous_encoder": [AEncoder(3,[8]), AEncoder(8,[16])],
            "encoder": PreTrain_AEncoder(3, pre3_checkpoint),
            "decoder": PreTrain_ADecoder(3),
            "is_variant": False,
            "previous_checkpoint": [r"D:\Paper\checkpoint_resnet_no_noise\2sae\encoder1.pth",
                                    r"D:\Paper\checkpoint_resnet_no_noise\2sae\encoder2.pth"],
            "checkpoint": [r"D:\Paper\checkpoint_resnet_no_noise\3sae\encoder1.pth",
                           r"D:\Paper\checkpoint_resnet_no_noise\3sae\encoder2.pth",
                           r"D:\Paper\checkpoint_resnet_no_noise\3sae\encoder3.pth",
                           r"D:\Paper\checkpoint_resnet_no_noise\3sae\decoder3.pth"]
        },
        "4SAE": {
            # "is_stack": True,
            # "previous_encoder": [AEncoder(3,[8]), AEncoder(8,[16]), AEncoder(16,[32])],
            "encoder": PreTrain_AEncoder(4, pre4_checkpoint),
            "decoder": PreTrain_ADecoder(4),
            "is_variant": False,
            "previous_checkpoint": [r"D:\Paper\checkpoint_resnet_no_noise\3sae\encoder1.pth",
                                    r"D:\Paper\checkpoint_resnet_noise\3sae\encoder2.pth",
                                    r"D:\Paper\checkpoint_resnet_noise\3sae\encoder3.pth"],
            "checkpoint": [r"D:\Paper\checkpoint_resnet_no_noise\4sae\encoder1.pth",
                           r"D:\Paper\checkpoint_resnet_no_noise\4sae\encoder2.pth",
                           r"D:\Paper\checkpoint_resnet_no_noise\4sae\encoder3.pth",
                           r"D:\Paper\checkpoint_resnet_no_noise\4sae\encoder4.pth",
                           r"D:\Paper\checkpoint_resnet_no_noise\4sae\decoder4.pth"]
        }
    }
}