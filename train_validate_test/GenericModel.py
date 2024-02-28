import torch
import numpy as np
import albumentations as A
import PIL

from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import TrainingArguments, Trainer
from datasets import DatasetDict, Metric
import evaluate

from typing import Any, Dict, List, Tuple, Optional
from LogClasses import WarningMessage, NoCheckpointException, NoTrainingDataException, NoTrainingArgumentException

class GenericModel:
    pass