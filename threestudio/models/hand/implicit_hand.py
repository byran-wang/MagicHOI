import threestudio
from threestudio.utils.base import BaseModule
from dataclasses import dataclass, field
from threestudio.utils.typing import *
from src.model.renderables.mano_node import MANONode

import os
import numpy as np

from src.model.mano.deformer import MANODeformer
from src.model.mano.server import MANOServer

@threestudio.register("implicit-hand")
class ImplicitHAND(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        data_dir: str = ""
        debug: bool = False
        log_dir: str = ""

    cfg: Config


    def configure(self) -> None:
        data_path = os.path.join(self.cfg.data_dir, f"build/data.npy")
        entities = np.load(data_path, allow_pickle=True).item()["entities"]

        betas_r = entities["right"]["mean_shape"]
    
        self.deformer = MANODeformer(max_dist=0.1, K=15, betas=betas_r, is_rhand=True)
        self.server = MANOServer(betas=betas_r, is_rhand=True)        