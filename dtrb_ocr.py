import os
import logging

import cv2
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

    def __init__(self, model_path, alphabet, imgW=100, use_gpu = False):
        self.using_gpu = use_gpu

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
                imgW
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
            s_prediction, num_class, imgW):
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
            "batch_max_length": 40,
        }

        return AttributeDict(options)

    def ocr_word(self, image):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img_w = self.options.imgH/image.shape[0]*image.shape[1]
        img_w = min(self.options.imgW, img_w)
        image = cv2.resize(image, (int(img_w),self.options.imgH))
        transformer = NormalizePAD((self.options.input_channel, self.options.imgH, self.options.imgW))


        image = Image.fromarray(image).convert('L')
        image = transformer(image)
        image = image.view(1, *image.size())
        if self.using_gpu:
            image = Variable(image).cuda()
        else:
            image = Variable(image)

        batch_size = 1
        length_for_pred = torch.IntTensor(
            [self.options.batch_max_length] * 1).to(self.device)
        text_for_pred = torch.LongTensor(
            1, self.options.batch_max_length + 1).fill_(0).to(self.device)

        if 'CTC' in self.options.Prediction:
            preds = self.model(image, text_for_pred).log_softmax(2)

            # Select max probabilty (greedy decoding) then decode index to alphabet
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
            preds_str = self.converter.decode(
                preds_index.data, preds_size.data)

        else:
            preds = self.model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to alphabet
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        if 'Attn' in self.options.Prediction:
            pred_EOS = preds_str[0].find('[s]')
            pred = preds_str[0][:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = preds_max_prob[0][:pred_EOS]
        else:
            pred_max_prob = preds_max_prob[0]
            pred = preds_str[0]
            
        # calculate confidence score (= multiply of pred_max_prob)
        confidence_score = float(pred_max_prob.cumprod(dim=0)[-1])

        return pred, confidence_score

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


        

