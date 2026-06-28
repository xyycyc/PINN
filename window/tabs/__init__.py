"""所有功能 Tab 的注册入口。"""

from .base import BaseCommandTab
from .build_db import BuildDbTab
from .path_settings import PathSettingsTab
from .train import TrainTab
from .predict import PredictTab
from .validate import ValidateTab
from .online_update import OnlineUpdateTab
from .demo import DemoTab
from .batch_modes import BatchModesTab
from .batch_preprocess import BatchPreprocessTab
from .search_weights import SearchWeightsTab
from .plot_loss import PlotLossTab
from .rerun_predict import RerunPredictTab
from .result_browser import ResultBrowserTab
from .manage_artifacts import ManageArtifactsTab

__all__ = [
    "BaseCommandTab",
    "PathSettingsTab",
    "BuildDbTab",
    "TrainTab",
    "PredictTab",
    "ValidateTab",
    "OnlineUpdateTab",
    "DemoTab",
    "BatchModesTab",
    "BatchPreprocessTab",
    "SearchWeightsTab",
    "PlotLossTab",
    "RerunPredictTab",
    "ResultBrowserTab",
    "ManageArtifactsTab",
]
