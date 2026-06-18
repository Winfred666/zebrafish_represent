"""Training framework package."""

from modules.framework.base import BaseTrainingFramework
from modules.framework.base_val import BaseValTrainingFramework
from modules.framework.IaN_flow import IaNFlowModule
from modules.framework.latent_ddpm import LatentDDPMModule
from modules.framework.mae import MAEFinetuneModule
from modules.framework.rect_flow import RectifiedFlowModule
from modules.framework.trellis_occupancy_vae import TRELLISOccupancyVAEModule
from modules.framework.vic_reg import VICRegModule
from modules.framework.vq_vae_s1 import VQVAES1Module
from modules.framework.vq_vae_s2 import VQVAES2Module

__all__ = [
    "BaseTrainingFramework",
    "BaseValTrainingFramework",
    "IaNFlowModule",
    "LatentDDPMModule",
    "MAEFinetuneModule",
    "RectifiedFlowModule",
    "TRELLISOccupancyVAEModule",
    "VICRegModule",
    "VQVAES1Module",
    "VQVAES2Module",
]
