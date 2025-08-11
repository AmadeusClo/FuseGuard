import torch
import torch.nn.functional as F
import torch.nn as nn
from .similar_utils import *
from copy import deepcopy

from .losses import mape_loss, mase_loss, smape_loss

loss_dict = {
    "l1": nn.L1Loss(),
    "smooth_l1": nn.SmoothL1Loss(),
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "smape": smape_loss(),
    "mape": mape_loss(),
    "mase": mase_loss(),
}


class cmLoss(nn.Module):
    def __init__(self, feature_loss, output_loss, task_loss, task_name, feature_w=0.01, output_w=1.0, task_w=1.0, trace_task_w=0.5, metric_task_w=0.5):
        """
        Cross-Modal Loss Function

        Args:
            feature_loss: Type of loss for feature regularization
            output_loss: Type of loss for output consistency
            task_loss: Type of loss for supervised task
            task_name: Name of the task
            feature_w: Weight for feature regularization loss (default: 0.01)
            output_w: Weight for output consistency loss (default: 1.0)
            task_w: Overall weight for task loss (default: 1.0)
            trace_task_w: Weight for trace modality task loss (default: 0.5)
            metric_task_w: Weight for metric modality task loss (default: 0.5)
          """

        super(cmLoss, self).__init__()
        # Main loss weights
        self.task_w = task_w  # Overall task loss weight
        self.output_w = output_w  # Output consistency weight
        self.feature_w = feature_w  # Feature regularization weight

        # Modality-specific task weights
        self.trace_task_w = trace_task_w  # Weight for trace modality
        self.metric_task_w = metric_task_w  # Weight for metric modality

        # Loss functions
        self.feature_loss = loss_dict[feature_loss]  # Feature regularization loss
        self.output_loss = loss_dict[output_loss]  # Output consistency loss
        self.task_loss = loss_dict[task_loss]  # Supervised task loss
        
        self.task_name = task_name

    def forward(self, outputs, batch_y_trace, batch_y_metric, in_sample=None, freq_map=None, batch_y_mark=None):
        # 1. Extract all outputs (according to the new return dictionary structure)
        outputs_time_trace = outputs["outputs_time_trace"]
        outputs_text_trace = outputs["outputs_text_trace"]
        outputs_time_metric = outputs["outputs_time_metric"]
        outputs_text_metric = outputs["outputs_text_metric"]

        # Extract intermediate features (now separated into trace and metric intermediate features)
        intermidiate_feat_time_trace = outputs["intermidiate_time_trace"]
        intermidiate_feat_text_trace = outputs["intermidiate_text_trace"]
        intermidiate_feat_time_metric = outputs["intermidiate_time_metric"]
        intermidiate_feat_text_metric = outputs["intermidiate_text_metric"]

        # 2. Feature regularization loss (now needs to calculate both trace and metric parts)
        feature_loss_trace = sum(
            [
                (0.8 ** idx) * self.feature_loss(feat_time, feat_text)
                for idx, (feat_time, feat_text) in enumerate(
                zip(intermidiate_feat_time_trace[::-1], intermidiate_feat_text_trace[::-1])
            )
            ]
        )

        feature_loss_metric = sum(
            [
                (0.8 ** idx) * self.feature_loss(feat_time, feat_text)
                for idx, (feat_time, feat_text) in enumerate(
                zip(intermidiate_feat_time_metric[::-1], intermidiate_feat_text_metric[::-1])
            )
            ]
        )
        feature_loss = (feature_loss_trace + feature_loss_metric) / 2  # Average or weighted sum

        # 3. Output consistency loss (calculate consistency between time and text outputs for both trace and metric)
        output_loss_trace = self.output_loss(outputs_time_trace, outputs_text_trace)
        output_loss_metric = self.output_loss(outputs_time_metric, outputs_text_metric)
        output_loss = (output_loss_trace + output_loss_metric) / 2

        # 4. Supervised task loss (calculate prediction error for both trace and metric)
        device = outputs_time_trace.device  # Unified device
        batch_y_trace = batch_y_trace.to(device)
        batch_y_metric = batch_y_metric.to(device)

        task_loss_trace = self.task_loss(outputs_time_trace, batch_y_trace)
        task_loss_metric = self.task_loss(outputs_time_metric, batch_y_metric)

        # Can adjust weights as needed (e.g., focus more on trace or metric predictions)
        task_loss = (self.trace_task_w * task_loss_trace +
                     self.metric_task_w * task_loss_metric)

        total_loss = self.task_w * task_loss + self.output_w * output_loss + self.feature_w * feature_loss
        return total_loss
