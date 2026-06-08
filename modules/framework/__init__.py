"""Training framework package."""

from modules.framework.base import BaseTrainingFramework
from modules.framework.IaN_flow import IaNFlowModule
from modules.framework.latent_ddpm import LatentDDPMModule
from modules.framework.mae import MAEFinetuneModule
from modules.framework.rect_flow import RectifiedFlowModule
from modules.framework.vic_reg import VICRegModule
from modules.framework.vq_vae_s1 import VQVAES1Module
from modules.framework.vq_vae_s2 import VQVAES2Module

__all__ = [
    "BaseTrainingFramework",
    "IaNFlowModule",
    "LatentDDPMModule",
    "MAEFinetuneModule",
    "RectifiedFlowModule",
    "VICRegModule",
    "VQVAES1Module",
    "VQVAES2Module",
]
