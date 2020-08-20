import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
from scipy.fftpack import dct, idct
from scipy import linalg as la
import torch.nn.functional as F
import logging

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    '''
    :param: int AvgPool2d_fre: This is the kernel size for AvgPool2d (for self.avgpool) towards frequency dimension.
                               Default is set to 10 assuming 80 dimensional fbank is given as input. While going through
                               layers from 1 to 4 (self.layer1 to self.layer4), the frequency dim gets down-sampled by 8.
                               So to make 80/8 = 10 numbers to one value, the freq dimension (kwargs['AvgPool2d_fre_ksize'])
                                in nn.AvgPool2d should be 10 (as default). It was 3 origianlly from Nanxin's code since he
                                used 23 dim mfcc/filter bank feature. (e.g. ceil(23/8) = 3)
    '''
    def __init__(self, block, layers, **kwargs):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        if 'AvgPool2d_fre_ksize' in kwargs:
            self.avgpool = nn.AvgPool2d((1, kwargs['AvgPool2d_fre_ksize']))
        else:
            self.avgpool = nn.AvgPool2d((1, 3)) # Nanxin's original code
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        #print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.shape)
        #x = self.maxpool(x)

        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)

        x = self.avgpool(x)
        #print(x.shape)
        x = x.view(x.size(0), x.size(1), x.size(2)).permute(0, 2, 1)

        return x

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

class LDE(nn.Module):
    def __init__(self, D, input_dim, with_bias=False, distance_type='norm', network_type='att', pooling='mean', regularization=None):
        super(LDE, self).__init__()
        self.dic = nn.Parameter(torch.randn(D, input_dim))
        nn.init.uniform_(self.dic.data, -1, 1)
        self.wei = nn.Parameter(torch.ones(D))
        if with_bias:
            self.bias = nn.Parameter(torch.zeros(D))
        else:
            self.bias = 0
        assert distance_type == 'norm' or distance_type == 'sqr'
        if distance_type == 'norm':
            self.dis = lambda x: torch.norm(x, p=2, dim=-1)
        else:
            self.dis = lambda x: torch.sum(x**2, dim=-1)
        assert network_type == 'att' or network_type == 'lde'
        if network_type == 'att':
            self.norm = lambda x: F.softmax(-self.dis(x) * self.wei + self.bias, dim = -2)
        else:
            self.norm = lambda x: F.softmax(-self.dis(x) * (self.wei ** 2) + self.bias, dim = -1)
        assert pooling == 'mean' or pooling == 'mean+std'
        self.pool = pooling
        if regularization is None:
            self.reg = None
        else:
            raise NotImplementedError()

    def forward(self, x):
        r = x.view(x.size(0), x.size(1), 1, x.size(2)) - self.dic
        w = self.norm(r).view(r.size(0), r.size(1), r.size(2), 1)
        w = w / (torch.sum(w, dim=1, keepdim=True) + 1e-9) #batch_size, timesteps, component
        if self.pool == 'mean':
            x = torch.sum(w * r, dim=1) 
        else:
            x1 = torch.sum(w * r, dim=1)
            x2 = torch.sqrt(torch.sum(w * r ** 2, dim=1)+1e-8)
            x = torch.cat([x1, x2], dim=-1)
        return x.view(x.size(0), -1)

# Model
class E2E_speakerid(nn.Module):
    def __init__(self, input_dim, output_dim, Q, D, hidden_dim=128, distance_type='norm', network_type='att', pooling='mean', regularization=None, asoftmax=False, resnet_AvgPool2d_fre_ksize = 3):
        super(E2E_speakerid, self).__init__()
        #self.lift = nn.Parameter(torch.from_numpy(1./_make_liftering(input_dim, Q)), requires_grad=False)
        #self.dct  = nn.Parameter(torch.from_numpy(_make_dct(input_dim, input_dim, inv=True, normalize=True)))
        self.res  = resnet34(AvgPool2d_fre_ksize = resnet_AvgPool2d_fre_ksize)
        self.pool = LDE(D, 128, distance_type=distance_type, network_type=network_type, pooling=pooling, regularization=regularization, with_bias=False)
        if pooling=='mean':
            self.fc1  = nn.Linear(128*D, hidden_dim)
        if pooling=='mean+std':
            self.fc1  = nn.Linear(256*D, hidden_dim)
        self.bn1  = nn.BatchNorm1d(hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, output_dim)
        logging.info("num speakers (spkid module's output dim): {}".format(output_dim))
        self.asoftmax = asoftmax

    def forward(self, x):
        #x = x * self.lift
        #x = F.linear(x, self.dct)
        x = self.res(x)
        x = self.pool(x)
        x = self.fc1(x)
        x_emb = self.bn1(x)
        if self.asoftmax:
            w = torch.transpose(self.fc2.weight, 0, 1) # size=(F,Classnum) F=in_features Classnum=out_features
            ww = w.renorm(2,1,1e-5).mul(1e5)
            xlen = x_emb.pow(2).sum(1).pow(0.5) # size=B
            wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum
            cos_theta = x_emb.mm(ww) # size=(B,Classnum)
            cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
            cos_theta = cos_theta.clamp(-1,1)
            self.mlambda = lambda x: 2*x**2-1
            cos_m_theta = self.mlambda(cos_theta)
            theta = torch.cuda.FloatTensor(cos_theta.data.acos())
            k = (2*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
            cos_theta = cos_theta * xlen.view(-1,1)
            phi_theta = phi_theta * xlen.view(-1,1)
            return x_emb, (cos_theta, phi_theta)
        else:
            x_emb = F.relu(x_emb)
            x_emb = self.fc2(x_emb)
            return F.log_softmax(x_emb, dim=-1)

    def predict(self, x):
        #x = x * self.lift
        #x = F.linear(x, self.dct)
        x = self.res(x)
        x = self.pool(x)
        if type(x) is tuple:
            x = x[0]
        x = self.fc1(x)
        x = self.bn1(x)
        return x

# Loss
class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte().detach()
        #index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.01*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        #logpt = F.log_softmax(output)
        logpt = F.log_softmax(output,dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp().detach()

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss

# decoding: generate speaker embeddings from a trained model
import json
import os
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.asr.asr_utils import get_model_conf
from espnet.utils.dynamic_import import dynamic_import
from espnet.nets.tts_interface import TTSInterface
from espnet.asr.asr_utils import torch_load
from utils.speakerid_kaldi_io import read_mat_scp
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.iterators import ToggleableShufflingSerialIterator
from chainer.datasets import TransformDataset
from espnet.tts.pytorch_backend.tts_speakerid import CustomConverter


def decode(args):
    """Decode with E2E-TTS model."""
    set_deterministic_pytorch(args)
    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # show arguments
    for key in sorted(vars(args).keys()):
        logging.info('args: ' + key + ': ' + str(vars(args)[key]))

    # define model
    model_class = dynamic_import(train_args.model_module)
    model = model_class(idim, odim, train_args)
    assert isinstance(model, TTSInterface)
    logging.info(model)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    torch_load(args.model, model)
    model.eval()

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)

    # generate speaker embeddings
    SequenceGenerator(model.resnet_spkid, device, args.feat_scp, args.out_file)

def SequenceGenerator(model, device, feat_scp, out_file):
    f = open(out_file, 'w')
    with torch.no_grad():
        for lab, x in read_mat_scp(feat_scp):
            y=[x]
            y_pred=model.predict(torch.from_numpy(np.array(y,dtype=np.float32)).to(device)).cpu().data.numpy().flatten()
            f.write(lab+' [ '+' '.join(map(str, y_pred.tolist()))+' ]\n')
    f.close()

#def classification_acc(model, ):
#    pred = spkid_out[0].max(1, keepdim=True)[1] # JJ (TODO) : currently values are ordered in a same way for both cos_theta and logp (Just following Nanxin's suggestion but need to check it)
#    correct = += pred.eq(spklabs.view_as(pred)).sum().item()
#    spkid_acc = correct/float(spklabs.shape[0])
#    return 
def eval_spkidclassification(args):
    """Decode with E2E-TTS model."""
    set_deterministic_pytorch(args)
    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # show arguments
    for key in sorted(vars(args).keys()):
        logging.info('args: ' + key + ': ' + str(vars(args)[key]))

    # define model
    model_class = dynamic_import(train_args.model_module)
    model = model_class(idim, odim, train_args)
    assert isinstance(model, TTSInterface)
    logging.info(model)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    torch_load(args.model, model)
    model.eval()

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)

    # read json data
    with open(args.json, 'rb') as f:
        valid_json = json.load(f)['utts']

    # define iteratior
    load_cv = LoadInputsAndTargets(
        mode='tts', sort_in_input_length=False,
        use_speaker_embedding=train_args.use_speaker_embedding,
        train_spkid_extractor=train_args.train_spkid_extractor,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )
    ### JJ - added
    ## iterator related
    valid_batchset = make_batchset(valid_json, args.batch_size,
                                   train_args.maxlen_in, train_args.maxlen_out, train_args.minibatches,
                                   batch_sort_key=train_args.batch_sort_key,
                                   min_batch_size=train_args.ngpu if train_args.ngpu > 1 else 1,
                                   count=train_args.batch_count,
                                   batch_bins=train_args.batch_bins,
                                   batch_frames_in=train_args.batch_frames_in,
                                   batch_frames_out=train_args.batch_frames_out,
                                   batch_frames_inout=train_args.batch_frames_inout,
                                   swap_io=True, iaxis=0, oaxis=0)
    valid_iter = ToggleableShufflingSerialIterator(
        TransformDataset(valid_batchset, load_cv),
        batch_size=1, repeat=False, shuffle=False)
    ## converter
    converter = CustomConverter()
    ### JJ - added end
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for batch in valid_iter:
            # convert to torch tensor
            x = converter(batch, device)
            if isinstance(x, tuple):
                num_correct, num_samples = model(*x, spk_clss_eval_on='on')
            else:
                num_correct, num_samples = model(**x, spk_clss_eval_on='on')
            total_correct += num_correct
            total_samples += num_samples
        actual_samples = sum(list(map(len,valid_batchset)))
        assert total_samples == actual_samples, "total samples: {} & actual samples: {}".format(total_samples, actual_samples)
        logging.warning("Classification accuracy on {} utterances: {}".format(total_samples, total_correct/float(total_samples)))
    #return total_correct, total_samples
