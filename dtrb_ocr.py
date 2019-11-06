import os
i

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from .utils import CTCLabelConverter, AttnLabelConverter
# from dataset import RawDataset, AlignCollate
from .model import Model

# the following is the alphabet, we're not supporting case-sensitive yet.
character = '0123456789abcdefghijklmnopqrstuvwxyz'

class DTRB_OCR:

    def __init__(self, model_path, use_gpu = False):
        self.using_gpu = use_gpu

        self.device = torch.device('cuda' if self.using_gpu else 'cpu')

        # IMPORTANT: will define a lot of params given the model_name
        filename, file_extension = os.path.splitext(os.path.basename(model_path))

        s_transformer, s_feature, s_sequence_model, s_prediction = filename.split("-")
        if 'CTC' in s_transformer:
            converter = CTCLabelConverter(character)
        else:
            converter = AttnLabelConverter(character)
        opt = get_default_options(s_transformer, s_feature, s_sequence_model, s_prediction, len(converter.character))
        
        self.model = Model(opt)
        self.model = torch.nn.DataParallel(self.model).to(self.device)

    def get_default_options(s_transformer, s_feature, s_sequence_model, s_prediction, num_class):
        options = {
            "Transformation": s_transformer,
            "FeatureExtraction": s_feature,
            "SequenceModeling": s_sequence_model,
            "Prediction": s_prediction,
            "num_fiducial": 20,
            "imgH": 32,
            "imgW": 100,
            "input_channel": 1,  # no RGB yet
            "output_channel": 512,
            "hidden_size": 256,
            "num_class": num_class,
            "batch_max_length": 25
        }

        return AttributeDict(options)


class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


        

