import os
import math
import logging

import cv2
import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from .utils import CTCLabelConverter, AttnLabelConverter
from .dataset import ResizeNormalize, NormalizePAD
from .model import Model

class DTRB_OCR:

    def __init__(self, model_path, alphabet, imgW=100, batch_max_length=40, PAD=False, use_gpu = False):
        self.using_gpu = use_gpu
        self.PAD = PAD

        self.device = torch.device('cuda' if self.using_gpu else 'cpu')

        # IMPORTANT: will define a lot of params given the model_name
        filename, file_extension = os.path.splitext(os.path.basename(model_path))

        s_transformer, s_feature, s_sequence_model, s_prediction = filename.split("-")[:4]
        logging.debug("s_transformer: "+str(s_transformer))
        if 'CTC' in s_prediction:
            self.converter = CTCLabelConverter(alphabet)
        else:
            self.converter = AttnLabelConverter(alphabet)
        self.options = self._get_default_options(
                s_transformer, s_feature, s_sequence_model,
                s_prediction, len(self.converter.character),
                imgW, batch_max_length
            )

        self.model = Model(self.options)
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        if self.using_gpu:
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        
        self.model.eval()

    def _get_default_options(
            self, s_transformer, s_feature, s_sequence_model,
            s_prediction, num_class, imgW, batch_max_length):
        options = {
            "Transformation": s_transformer,
            "FeatureExtraction": s_feature,
            "SequenceModeling": s_sequence_model,
            "Prediction": s_prediction,
            "num_fiducial": 20,
            "imgH": 32,
            "imgW": imgW,
            "input_channel": 1,  # no RGB yet
            "output_channel": 512,
            "hidden_size": 256,
            "num_class": num_class,
            "batch_max_length": batch_max_length,
        }

        return AttributeDict(options)

    def ocr_batch(self, images):
        for i in range(len(images)):
            if len(images[i].shape) > 2:
                images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            if images[i].shape[0] != self.options.imgH:
                if self.PAD:
                    scale = self.options.imgH/images[i].shape[0]
                    W = min(self.options.imgW, math.ceil(images[i].shape[1]*scale))
                else:
                    W = self.options.imgW
                tmp_img = Image.fromarray(images[i])
                tmp_img = tmp_img.resize((W, self.options.imgH), Image.BICUBIC)
                images[i] = np.array(tmp_img)
            if self.PAD:
                images[i] = np.pad(
                        images[i],
                        ((0,0),(0,max(0,self.options.imgW-images[i].shape[1]))),
                        'edge')
                images[i] = images[i].reshape((1,self.options.imgH, self.options.imgW))
            images[i] = (images[i].astype(np.float32)/255.0-0.5)*2.0

        image_batch = np.asarray(images)
        image = torch.from_numpy(image_batch).to(self.device)
        batch_size = len(images)
        # For max length prediction
        length_for_pred = torch.IntTensor([self.options.batch_max_length] * batch_size).to(self.device)
        text_for_pred = torch.LongTensor(batch_size, self.options.batch_max_length + 1).fill_(0).to(self.device)

        if 'CTC' in self.options.Prediction:
            preds = self.model(image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
            preds_str = self.converter.decode(preds_index.data, preds_size.data)

        else:
            preds = self.model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        conf_list, pred_list = [], []
        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            if 'Attn' in self.options.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            conf_list.append(float(pred_max_prob.cumprod(dim=0)[-1]))
            pred_list.append(pred)

        return pred_list, conf_list


    def ocr_word(self, image):
        preds, confs = self.ocr_batch([image])
        return preds[0], confs[0]

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


        

