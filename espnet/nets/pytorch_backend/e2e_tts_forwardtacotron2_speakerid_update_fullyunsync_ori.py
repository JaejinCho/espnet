#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# JJ: edited from espnet/nets/pytorch_backend/e2e_tts_forwardtacotron2_speakerid_update_unsync.py
# JJ: edited from espnet/nets/pytorch_backend/e2e_tts_forwardtacotron2_speakerid_update.py
# JJ: edited from espnet/nets/pytorch_backend/e2e_tts_tacotron2_speakerid_update.py
# JJ: edited from espnet/nets/pytorch_backend/e2e_tts_tacotron2_speakerid.py
# to covoer Decoder_noatt in espnet/nets/pytorch_backend/tacotron2/decoder_update.py
# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Tacotron 2 related modules."""

import logging

from distutils.util import strtobool

import numpy as np
import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.rnn.attentions import AttForward
from espnet.nets.pytorch_backend.rnn.attentions import AttForwardTA
from espnet.nets.pytorch_backend.rnn.attentions import AttLoc
from espnet.nets.pytorch_backend.tacotron2.cbhg import CBHG
from espnet.nets.pytorch_backend.tacotron2.cbhg import CBHGLoss
from espnet.nets.pytorch_backend.tacotron2.decoder import Decoder
from espnet.nets.pytorch_backend.tacotron2.decoder_update import Decoder_noatt
from espnet.nets.pytorch_backend.tacotron2.decoder_update_forwardtacotron2 import Decoder_forward
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.fill_missing_args import fill_missing_args
from utils.speakerid import E2E_speakerid, AngleLoss


class GuidedAttentionLoss(torch.nn.Module):
    """Guided attention loss function module.

    This module calculates the guided attention loss described in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_, which forces the attention to be diagonal.

    .. _`Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969

    """

    def __init__(self, sigma=0.4, alpha=1.0, reset_always=True):
        """Initialize guided attention loss module.

        Args:
            sigma (float, optional): Standard deviation to control how close attention to a diagonal.
            alpha (float, optional): Scaling coefficient (lambda).
            reset_always (bool, optional): Whether to always reset masks.

        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def forward(self, att_ws, ilens, olens):
        """Calculate forward propagation.

        Args:
            att_ws (Tensor): Batch of attention weights (B, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).

        Returns:
            Tensor: Guided attention loss value.

        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(att_ws.device)
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()
        return self.alpha * loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(ilen, olen, self.sigma)
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        """Make guided attention mask.

        Examples:
            >>> guided_attn_mask =_make_guided_attention(5, 5, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([5, 5])
            >>> guided_attn_mask
            tensor([[0.0000, 0.1175, 0.3935, 0.6753, 0.8647],
                    [0.1175, 0.0000, 0.1175, 0.3935, 0.6753],
                    [0.3935, 0.1175, 0.0000, 0.1175, 0.3935],
                    [0.6753, 0.3935, 0.1175, 0.0000, 0.1175],
                    [0.8647, 0.6753, 0.3935, 0.1175, 0.0000]])
            >>> guided_attn_mask =_make_guided_attention(3, 6, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([6, 3])
            >>> guided_attn_mask
            tensor([[0.0000, 0.2934, 0.7506],
                    [0.0831, 0.0831, 0.5422],
                    [0.2934, 0.0000, 0.2934],
                    [0.5422, 0.0831, 0.0831],
                    [0.7506, 0.2934, 0.0000],
                    [0.8858, 0.5422, 0.0831]])

        """
        grid_x, grid_y = torch.meshgrid(torch.arange(olen), torch.arange(ilen))
        grid_x, grid_y = grid_x.float(), grid_y.float()
        return 1.0 - torch.exp(-(grid_y / ilen - grid_x / olen) ** 2 / (2 * (sigma ** 2)))

    @staticmethod
    def _make_masks(ilens, olens):
        """Make masks indicating non-padded part.

        Examples:
            >>> ilens, olens = [5, 2], [8, 5]
            >>> _make_mask(ilens, olens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)

        """
        in_masks = make_non_pad_mask(ilens)  # (B, T_in)
        out_masks = make_non_pad_mask(olens)  # (B, T_out)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)


class Tacotron2Loss(torch.nn.Module):
    """Loss function module for Tacotron2."""

    def __init__(self, use_masking=True, bce_pos_weight=20.0):
        """Initialize Tactoron2 loss module.

        Args:
            use_masking (bool): Whether to mask padded part in loss calculation.
            bce_pos_weight (float): Weight of positive sample of stop token.

        """
        super(Tacotron2Loss, self).__init__()
        self.use_masking = use_masking
        self.bce_pos_weight = bce_pos_weight

    def forward(self, after_outs, before_outs, logits, ys, labels, olens):
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            logits (Tensor): Batch of stop logits (B, Lmax).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            labels (LongTensor): Batch of the sequences of stop token labels (B, Lmax).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Mean square error loss value.
            Tensor: Binary cross entropy loss value.

        """
        # perform masking for padded values
        if self.use_masking:
            mask = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            ys = ys.masked_select(mask)
            after_outs = after_outs.masked_select(mask)
            before_outs = before_outs.masked_select(mask)
            labels = labels.masked_select(mask[:, :, 0])
            logits = logits.masked_select(mask[:, :, 0])

        # calculate loss
        l1_loss = F.l1_loss(after_outs, ys) + F.l1_loss(before_outs, ys)
        mse_loss = F.mse_loss(after_outs, ys) + F.mse_loss(before_outs, ys)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=torch.tensor(self.bce_pos_weight, device=ys.device))

        return l1_loss, mse_loss, bce_loss


class Tacotron2(TTSInterface, torch.nn.Module):
    """Tacotron2 module for end-to-end text-to-speech (E2E-TTS).

    This is a module of Spectrogram prediction network in Tacotron2 described in `Natural TTS Synthesis
    by Conditioning WaveNet on Mel Spectrogram Predictions`_, which converts the sequence of characters
    into the sequence of Mel-filterbanks.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    @staticmethod
    def add_arguments(parser):
        """Add model-specific arguments to the parser."""
        group = parser.add_argument_group("tacotron 2 model setting")
        # encoder
        group.add_argument('--embed-dim', default=512, type=int,
                           help='Number of dimension of embedding')
        group.add_argument('--elayers', default=1, type=int,
                           help='Number of encoder layers')
        group.add_argument('--eunits', '-u', default=512, type=int,
                           help='Number of encoder hidden units')
        group.add_argument('--econv-layers', default=3, type=int,
                           help='Number of encoder convolution layers')
        group.add_argument('--econv-chans', default=512, type=int,
                           help='Number of encoder convolution channels')
        group.add_argument('--econv-filts', default=5, type=int,
                           help='Filter size of encoder convolution')
        # attention
        group.add_argument('--atype', default="location", type=str,
                           choices=["noatt","forward_ta", "forward", "location"],
                           help='Type of attention mechanism')
        group.add_argument('--adim', default=512, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aconv-chans', default=32, type=int,
                           help='Number of attention convolution channels')
        group.add_argument('--aconv-filts', default=15, type=int,
                           help='Filter size of attention convolution')
        group.add_argument('--cumulate-att-w', default=True, type=strtobool,
                           help="Whether or not to cumulate attention weights")
        # decoder
        group.add_argument('--dlayers', default=2, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=1024, type=int,
                           help='Number of decoder hidden units')
        group.add_argument('--prenet-layers', default=2, type=int,
                           help='Number of prenet layers')
        group.add_argument('--prenet-units', default=256, type=int,
                           help='Number of prenet hidden units')
        group.add_argument('--postnet-layers', default=5, type=int,
                           help='Number of postnet layers')
        group.add_argument('--postnet-chans', default=512, type=int,
                           help='Number of postnet channels')
        group.add_argument('--postnet-filts', default=5, type=int,
                           help='Filter size of postnet')
        group.add_argument('--output-activation', default=None, type=str, nargs='?',
                           help='Output activation function')
        group.add_argument('--rnntype', default='lstm', type=str,
                           choices=['lstm','gru'],
                           help='rnntype in decoder')
        group.add_argument('--bidirection', default=True, type=strtobool,
                           help='Whether to use bidirectional rnn or not')
        # cbhg
        group.add_argument('--use-cbhg', default=False, type=strtobool,
                           help='Whether to use CBHG module')
        group.add_argument('--cbhg-conv-bank-layers', default=8, type=int,
                           help='Number of convoluional bank layers in CBHG')
        group.add_argument('--cbhg-conv-bank-chans', default=128, type=int,
                           help='Number of convoluional bank channles in CBHG')
        group.add_argument('--cbhg-conv-proj-filts', default=3, type=int,
                           help='Filter size of convoluional projection layer in CBHG')
        group.add_argument('--cbhg-conv-proj-chans', default=256, type=int,
                           help='Number of convoluional projection channels in CBHG')
        group.add_argument('--cbhg-highway-layers', default=4, type=int,
                           help='Number of highway layers in CBHG')
        group.add_argument('--cbhg-highway-units', default=128, type=int,
                           help='Number of highway units in CBHG')
        group.add_argument('--cbhg-gru-units', default=256, type=int,
                           help='Number of GRU units in CBHG')
        # model (parameter) related
        group.add_argument('--use-batch-norm', default=True, type=strtobool,
                           help='Whether to use batch normalization')
        group.add_argument('--use-concate', default=True, type=strtobool,
                           help='Whether to concatenate encoder embedding with decoder outputs')
        group.add_argument('--use-residual', default=True, type=strtobool,
                           help='Whether to use residual connection in conv layer')
        group.add_argument('--dropout-rate', default=0.5, type=float,
                           help='Dropout rate')
        group.add_argument('--zoneout-rate', default=0.1, type=float,
                           help='Zoneout rate')
        group.add_argument('--reduction-factor', default=1, type=int,
                           help='Reduction factor')
        group.add_argument('--frame-subsampling-factor', default=3, type=int,
                           help='Frame subsampling factor (in Kaldi ASR)')
        group.add_argument("--spk-embed-dim", default=None, type=int,
                           help="Number of speaker embedding dimensions")
        group.add_argument("--spc-dim", default=None, type=int,
                           help="Number of spectrogram dimensions")
        # loss related
        group.add_argument('--use-masking', default=False, type=strtobool,
                           help='Whether to use masking in calculation of loss')
        group.add_argument('--bce-pos-weight', default=20.0, type=float,
                           help='Positive sample weight in BCE calculation (only for use-masking=True)')
        group.add_argument("--use-guided-attn-loss", default=False, type=strtobool,
                           help="Whether to use guided attention loss")
        group.add_argument("--guided-attn-loss-sigma", default=0.4, type=float,
                           help="Sigma in guided attention loss")
        group.add_argument("--guided-attn-loss-lambda", default=1.0, type=float,
                           help="Lambda in guided attention loss")
        return parser

    def __init__(self, idim, odim, args=None):
        """Initialize Tacotron2 module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            args (Namespace, optional):
                - embed_dim (int): Dimension of character embedding.
                - elayers (int): The number of encoder blstm layers.
                - eunits (int): The number of encoder blstm units.
                - econv_layers (int): The number of encoder conv layers.
                - econv_filts (int): The number of encoder conv filter size.
                - econv_chans (int): The number of encoder conv filter channels.
                - dlayers (int): The number of decoder lstm layers.
                - dunits (int): The number of decoder lstm units.
                - prenet_layers (int): The number of prenet layers.
                - prenet_units (int): The number of prenet units.
                - postnet_layers (int): The number of postnet layers.
                - postnet_filts (int): The number of postnet filter size.
                - postnet_chans (int): The number of postnet filter channels.
                - output_activation (int): The name of activation function for outputs.
                - adim (int): The number of dimension of mlp in attention.
                - aconv_chans (int): The number of attention conv filter channels.
                - aconv_filts (int): The number of attention conv filter size.
                - cumulate_att_w (bool): Whether to cumulate previous attention weight.
                - use_batch_norm (bool): Whether to use batch normalization.
                - use_concate (int): Whether to concatenate encoder embedding with decoder lstm outputs.
                - dropout_rate (float): Dropout rate.
                - zoneout_rate (float): Zoneout rate.
                - reduction_factor (int): Reduction factor.
                - spk_embed_dim (int): Number of speaker embedding dimension.
                - spc_dim (int): Number of spectrogram embedding dimenstions (only for use_cbhg=True).
                - use_cbhg (bool): Whether to use CBHG module.
                - cbhg_conv_bank_layers (int): The number of convoluional banks in CBHG.
                - cbhg_conv_bank_chans (int): The number of channels of convolutional bank in CBHG.
                - cbhg_proj_filts (int): The number of filter size of projection layeri in CBHG.
                - cbhg_proj_chans (int): The number of channels of projection layer in CBHG.
                - cbhg_highway_layers (int): The number of layers of highway network in CBHG.
                - cbhg_highway_units (int): The number of units of highway network in CBHG.
                - cbhg_gru_units (int): The number of units of GRU in CBHG.
                - use_masking (bool): Whether to mask padded part in loss calculation.
                - bce_pos_weight (float): Weight of positive sample of stop token (only for use_masking=True).
                - use-guided-attn-loss (bool): Whether to use guided attention loss.
                - guided-attn-loss-sigma (float) Sigma in guided attention loss.
                - guided-attn-loss-lamdba (float): Lambda in guided attention loss.

        """
        # initialize base classes
        TTSInterface.__init__(self)
        torch.nn.Module.__init__(self)

        # fill missing arguments
        args = fill_missing_args(args, self.add_arguments)

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.spk_embed_dim = args.spk_embed_dim
        self.spkloss_weight = args.spkloss_weight
        self.cumulate_att_w = args.cumulate_att_w
        self.reduction_factor = args.reduction_factor
        self.frame_subsampling_factor = args.frame_subsampling_factor
        self.use_cbhg = args.use_cbhg
        self.use_guided_attn_loss = args.use_guided_attn_loss

        # define activation function for the final output
        if args.output_activation is None:
            self.output_activation_fn = None
        elif hasattr(F, args.output_activation):
            self.output_activation_fn = getattr(F, args.output_activation)
        else:
            raise ValueError('there is no such an activation function. (%s)' % args.output_activation)

        # set padding idx
        padding_idx = 0

        # define network modules
        self.enc = Encoder(idim=idim,
                           embed_dim=args.embed_dim,
                           elayers=args.elayers,
                           eunits=args.eunits,
                           econv_layers=args.econv_layers,
                           econv_chans=args.econv_chans,
                           econv_filts=args.econv_filts,
                           use_batch_norm=args.use_batch_norm,
                           use_residual=args.use_residual,
                           dropout_rate=args.dropout_rate,
                           padding_idx=padding_idx)
        dec_idim = args.eunits if args.spk_embed_dim is None else args.eunits + args.spk_embed_dim
        # JJ added - start: TODO(Try to understand input parameters for E2E_speakerid)
        if args.train_spkid_extractor:
            self.train_spkid_extractor = True
            self.resnet_spkid = E2E_speakerid(input_dim=odim, output_dim=args.num_spk, Q=odim-1, D=32, hidden_dim=args.spk_embed_dim, pooling='mean',
                        network_type='lde', distance_type='sqr', asoftmax=True, resnet_AvgPool2d_fre_ksize=10)
            self.angle_loss = AngleLoss()
        # JJ added - end
        if args.atype == "location":
            att = AttLoc(dec_idim,
                         args.dunits,
                         args.adim,
                         args.aconv_chans,
                         args.aconv_filts)
        elif args.atype == "forward":
            att = AttForward(dec_idim,
                             args.dunits,
                             args.adim,
                             args.aconv_chans,
                             args.aconv_filts)
            if self.cumulate_att_w:
                logging.warning("cumulation of attention weights is disabled in forward attention.")
                self.cumulate_att_w = False
        elif args.atype == "forward_ta":
            att = AttForwardTA(dec_idim,
                               args.dunits,
                               args.adim,
                               args.aconv_chans,
                               args.aconv_filts,
                               odim)
            if self.cumulate_att_w:
                logging.warning("cumulation of attention weights is disabled in forward attention.")
                self.cumulate_att_w = False
        elif args.atype == "noatt":
            att = None
        else:
            raise NotImplementedError("Support only location or forward")

        if att == None:
            self.dec = Decoder_forward(idim=dec_idim,
                               odim=odim,
                               dlayers=args.dlayers,
                               dunits=args.dunits,
                               prenet_layers=args.prenet_layers,
                               prenet_units=args.prenet_units,
                               postnet_layers=args.postnet_layers,
                               postnet_chans=args.postnet_chans,
                               postnet_filts=args.postnet_filts,
                               output_activation_fn=self.output_activation_fn,
                               use_batch_norm=args.use_batch_norm,
                               use_concate=args.use_concate,
                               dropout_rate=args.dropout_rate,
                               zoneout_rate=args.zoneout_rate,
                               reduction_factor=args.reduction_factor,
                               frame_subsampling_factor=args.frame_subsampling_factor,
                               rnntype=args.rnntype,
                               bidirection=args.bidirection)
        else:
            self.dec = Decoder(idim=dec_idim,
                               odim=odim,
                               att=att,
                               dlayers=args.dlayers,
                               dunits=args.dunits,
                               prenet_layers=args.prenet_layers,
                               prenet_units=args.prenet_units,
                               postnet_layers=args.postnet_layers,
                               postnet_chans=args.postnet_chans,
                               postnet_filts=args.postnet_filts,
                               output_activation_fn=self.output_activation_fn,
                               cumulate_att_w=self.cumulate_att_w,
                               use_batch_norm=args.use_batch_norm,
                               use_concate=args.use_concate,
                               dropout_rate=args.dropout_rate,
                               zoneout_rate=args.zoneout_rate,
                               reduction_factor=args.reduction_factor)
        self.taco2_loss = Tacotron2Loss(use_masking=args.use_masking,
                                        bce_pos_weight=args.bce_pos_weight)
        if self.use_guided_attn_loss:
            self.attn_loss = GuidedAttentionLoss(
                sigma=args.guided_attn_loss_sigma,
                alpha=args.guided_attn_loss_lambda,
            )
        if self.use_cbhg:
            self.cbhg = CBHG(idim=odim,
                             odim=args.spc_dim,
                             conv_bank_layers=args.cbhg_conv_bank_layers,
                             conv_bank_chans=args.cbhg_conv_bank_chans,
                             conv_proj_filts=args.cbhg_conv_proj_filts,
                             conv_proj_chans=args.cbhg_conv_proj_chans,
                             highway_layers=args.cbhg_highway_layers,
                             highway_units=args.cbhg_highway_units,
                             gru_units=args.cbhg_gru_units)
            self.cbhg_loss = CBHGLoss(use_masking=args.use_masking)

    def forward(self, xs, ilens, ys, labels, olens, spembs=None, spcs=None, spklabs=None, *args, **kwargs):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of padded character ids (B, Tmax).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional): Batch of speaker embedding vectors (B, spk_embed_dim).
            spcs (Tensor, optional): Batch of groundtruth spectrograms (B, Lmax, spc_dim).

        Returns:
            Tensor: Loss value.

        """
        # remove unnecessary padded part (for multi-gpus)
        max_in = max(ilens)
        max_out = max(olens)
        if max_in != xs.shape[1]:
            xs = xs[:, :max_in]
        if max_out != ys.shape[1]:
            ys = ys[:, :max_out]
            labels = labels[:, :max_out]

        # (TODO) Check requires_grad
        if self.training == True:
            # 1. get random chunk length. Try: 1) multiple ranges e.g.) 200 to 400, 300 to 500, etc) 2) fixed length
            bs = xs.shape[0] # batch size
            chunk_len = int(torch.randint(200,401,(1,)))
            #ys_4spembs = torch.zeros([bs,chunk_len,ys.shape[-1]]).to(ys.device) # (TODO: DONE a line below) This part should change to reduce zero-padding (refer to e2e_tts_tacotron2_speakerid_update_fullyunsync.py
            ys_4spembs = torch.zeros([bs, max_out if max_out < chunk_len else chunk_len, ys.shape[-1]]).to(ys.device) # if statement here to minimize zero-padding
            # 2. get a random chunk per sample with zero-padding
            for i in range(bs):
                if int(olens[i]) > chunk_len:
                    max_start = int(olens[i]) - chunk_len
                    start_ix = int(torch.randint(0,max_start+1,(1,)))
                    ys_4spembs[i] = ys[i,start_ix:start_ix+chunk_len]
                else:
                    ys_4spembs[i,:int(olens[i])] = ys[i,:int(olens[i])] # or ys_4spembs[i] = ys[i,:chunk_len]
            if ys.requires_grad == True:
                ys_4spembs.requires_grad = True
            ## 2) For target frames (ys_4phnali) and phone ali (xs_4ys_4phnali)
            # self.reduction_factor (=3) * len phn ali > len ys by 1, 2 or 3
            # (when frame_subsample_rate in Kaldi = 3). That being said, I
            # think it is okay to have a little longer phn ali than ys as above 
            # thus *** I first make length of ys as multiples of 3 (by adding
            # 1, 2 optionally) and divide it by 3 so that I
            # could get length of phn ali
            chunk_len = int(torch.randint(200,401,(1,)))
            ys_4phnali = torch.zeros([bs, max_out if max_out < chunk_len else chunk_len, ys.shape[-1]]).to(ys.device) # if statement here to minimize zero-padding. If the size of ys_4phnali is not properly determined, this could cause some errors as I confronted at the taco2loss calculation due to dimension mismatch
            # In ForwardTacotron2, self.reduction_factor == 1 and
            # self.reduction_factor in Tacotron2 corresponds to
            # self.frame_subsampling_factor most of the time
            if (chunk_len % self.frame_subsampling_factor) == 0:
                chunk_len_phn = int(chunk_len/self.frame_subsampling_factor)
            else:
                chunk_len_phn = int((chunk_len - chunk_len % self.frame_subsampling_factor)/self.frame_subsampling_factor + 1) # same as the line above but easier to understand (+1 in the end because 3 * len(phn_ali.s) should be larger than len(ys) due to the way the code is written in decoder
            
            xs_4ys_4phnali = torch.zeros([bs, max_in - 1 if max_in - 1 < chunk_len_phn else chunk_len_phn], dtype=torch.long).to(ys.device) # max_in - 1 here since xs is padded with eos in the ends
            ilens_chunk = torch.zeros([bs], dtype=torch.long).to(ys.device)
            olens_chunk = torch.zeros([bs], dtype=torch.long).to(ys.device)
            for i in range(bs):
                if int(olens[i]) > chunk_len:
                    max_start = int(olens[i]) - chunk_len
                    # start_ix is now multiples of 3 (including 0)
                    rint = int(torch.randint(0,max_start+1,(1,)))
                    if (rint % self.frame_subsampling_factor) == 0:
                        start_ix = rint
                    else:
                        start_ix = rint - rint % self.frame_subsampling_factor
                    ys_4phnali[i] = ys[i,start_ix:start_ix+chunk_len]
                    xs_4ys_4phnali[i] = xs[i,int(start_ix/3):int(start_ix/3)+chunk_len_phn] # ******** NOTE *******: currently, eos is not added for chunks. I am not sure if it is okay but later put extra care to code this part
                    ilens_chunk[i] = chunk_len_phn
                    olens_chunk[i] = chunk_len
                else:
                    ys_4phnali[i,:int(olens[i])] = ys[i,:int(olens[i])]
                    #xs_4ys_4phnali[i,:ilens[i]-1] = xs[i,:ilens[i]-1] # ******** NOTE *******: currently, eos is not added for chunks. I am not sure if it is okay but later put extra care to code this part (original xs includes the eos at the ends)
                    if (olens[i] % self.frame_subsampling_factor) == 0:
                        len_phn = int(olens[i]/self.frame_subsampling_factor)
                    else:
                        len_phn = int((olens[i] - olens[i] % self.frame_subsampling_factor)/self.frame_subsampling_factor + 1)
                    xs_4ys_4phnali[i,:len_phn] = xs[i,:len_phn] # ******** NOTE *******: currently, eos is not added for chunks. I am not sure if it is okay but later put extra care to code this part (original xs includes the eos at the ends)
                    ilens_chunk[i] = len_phn
                    olens_chunk[i] = int(olens[i])
            if ys.requires_grad == True:
                ys_4phnali.requires_grad = True
            if xs.requires_grad == True:
                xs_4ys_4phnali.requires_grad = True
            hs, hlens = self.enc(xs_4ys_4phnali, ilens_chunk)
        else: # during eval
            hs, hlens = self.enc(xs, ilens)
        # JJ added - end

        # calculate tacotron2 outputs
        if self.spk_embed_dim is not None:
            if self.train_spkid_extractor:
                if self.training == True:
                    spembs, spkid_out = self.resnet_spkid(ys_4spembs) # Currently working with a batch padded zero-padded sequence vectors (Nanxin did not use it so the current spk id model does not do any specific things to deal with the zero-padding)
                else: # model.eval()
                    spembs, spkid_out = self.resnet_spkid(ys)
                spembs = spembs.unsqueeze(1).expand(-1, hs.size(1), -1)
            else:
                spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = torch.cat([hs, spembs], dim=-1)
        if self.training == True:
            after_outs, before_outs, logits, att_ws = self.dec(hs, hlens, ys_4phnali)
            # modifiy mod part of groundtruth (JJ: This is NOT needed for my FT2
            # since the predicted fbank seq lengths exactly follows olens (the
            # process happens in decoder module using len_cut var)
            #if self.reduction_factor > 1:
            #    olens_chunk = olens_chunk.new([olen_chunk - olen_chunk % self.reduction_factor for olen_chunk in olens_chunk])
            #    max_out = max(olens_chunk)
            #    ys_4phnali = ys_4phnali[:, :max_out]
            #    labels = labels[:, :max_out]
            #    labels[:, -1] = 1.0  # make sure at least one frame has 1

            # caluculate taco2 loss
            labels = labels[:, :ys_4phnali.size(1)]
            labels[:, -1] = 1.0
            l1_loss, mse_loss, bce_loss = self.taco2_loss(
                after_outs, before_outs, logits, ys_4phnali, labels, olens_chunk)
        else:
            after_outs, before_outs, logits, att_ws = self.dec(hs, hlens, ys)
            # modifiy mod part of groundtruth (JJ: This is NOT needed for my FT2
            # since the predicted fbank seq lengths exactly follows olens (the
            # process happens in decoder module using len_cut var)
            #if self.reduction_factor > 1:
            #    olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            #    max_out = max(olens)
            #    ys = ys[:, :max_out]
            #    labels = labels[:, :max_out]
            #    labels[:, -1] = 1.0  # make sure at least one frame has 1

            # caluculate taco2 loss
            l1_loss, mse_loss, bce_loss = self.taco2_loss(
                after_outs, before_outs, logits, ys, labels, olens)
        ## loss
        spkid_loss = self.angle_loss(spkid_out, spklabs) # JJ: spklabs are ground truth labels for speaker id
        loss = l1_loss + mse_loss + bce_loss + self.spkloss_weight * spkid_loss
        ## acc
        pred = spkid_out[0].max(1, keepdim=True)[1] # JJ (TODO) : currently values are ordered in a same way for both cos_theta and logp (Just following Nanxin's suggestion but need to check it)
        correct = pred.eq(spklabs.view_as(pred)).sum().item()
        spkid_acc = correct/float(spklabs.shape[0])
        report_keys = [
            {'l1_loss': l1_loss.item()},
            {'mse_loss': mse_loss.item()},
            {'bce_loss': bce_loss.item()},
            {'spkid_loss': spkid_loss.item()},
            {'spkid_acc': spkid_acc }
        ]

        # caluculate attention loss (JJ: This does NOT happen in FT2 due to NO att)
        if self.use_guided_attn_loss:
            # NOTE(kan-bayashi): length of output for auto-regressive input will be changed when r > 1
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            attn_loss = self.attn_loss(att_ws, ilens, olens_in)
            loss = loss + attn_loss
            report_keys += [
                {'attn_loss': attn_loss.item()},
            ]

        # caluculate cbhg loss
        if self.use_cbhg:
            # remove unnecessary padded part (for multi-gpus)
            if max_out != spcs.shape[1]:
                spcs = spcs[:, :max_out]

            # caluculate cbhg outputs & loss and report them
            cbhg_outs, _ = self.cbhg(after_outs, olens)
            cbhg_l1_loss, cbhg_mse_loss = self.cbhg_loss(cbhg_outs, spcs, olens)
            loss = loss + cbhg_l1_loss + cbhg_mse_loss
            report_keys += [
                {'cbhg_l1_loss': cbhg_l1_loss.item()},
                {'cbhg_mse_loss': cbhg_mse_loss.item()},
            ]

        report_keys += [{'loss': loss.item()}]
        self.reporter.report(report_keys)

        return loss

    def inference(self, x, inference_args, spemb=None, *args, **kwargs):
        """Generate the sequence of features given the sequences of characters.

        Args:
            x (Tensor): Input sequence of characters (T,).
            inference_args (Namespace):
                - threshold (float): Threshold in inference.
                - minlenratio (float): Minimum length ratio in inference.
                - maxlenratio (float): Maximum length ratio in inference.
            spemb (Tensor, optional): Speaker embedding vector (spk_embed_dim).

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Attention weights (L, T).

        """
        # get options
        threshold = inference_args.threshold
        minlenratio = inference_args.minlenratio
        maxlenratio = inference_args.maxlenratio

        # inference
        h = self.enc.inference(x)
        if self.spk_embed_dim is not None:
            if inference_args.train_spkid_extractor:
                spemb = spemb.expand(h.size(0), -1)
            else:
                spemb = F.normalize(spemb, dim=0).unsqueeze(0).expand(h.size(0), -1)
            h = torch.cat([h, spemb], dim=-1)
        outs, probs, att_ws = self.dec.inference(h, threshold, minlenratio, maxlenratio)

        if self.use_cbhg:
            cbhg_outs = self.cbhg.inference(outs)
            return cbhg_outs, probs, att_ws
        else:
            return outs, probs, att_ws

    def calculate_all_attentions(self, xs, ilens, ys, spembs=None, *args, **kwargs):
        """Calculate all of the attention weights.

        Args:
            xs (Tensor): Batch of padded character ids (B, Tmax).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional): Batch of speaker embedding vectors (B, spk_embed_dim).

        Returns:
            numpy.ndarray: Batch of attention weights (B, Lmax, Tmax).

        """
        # check ilens type (should be list of int)
        if isinstance(ilens, torch.Tensor) or isinstance(ilens, np.ndarray):
            ilens = list(map(int, ilens))

        self.eval()
        with torch.no_grad():
            hs, hlens = self.enc(xs, ilens)
            if self.spk_embed_dim is not None:
                if self.train_spkid_extractor:
                    spembs, spkid_out = self.resnet_spkid(ys) # Currently working with a batch padded zero-padded sequence vectors (Nanxin did not use it so the current spk id model does not do any specific things to deal with the zero-padding)
                    spembs = spembs.unsqueeze(1).expand(-1, hs.size(1), -1)
                else:
                    spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
                hs = torch.cat([hs, spembs], dim=-1)
            att_ws = self.dec.calculate_all_attentions(hs, hlens, ys)
        self.train()

        return att_ws.cpu().numpy()

    @property
    def base_plot_keys(self):
        """Return base key names to plot during training. keys should match what `chainer.reporter` reports.

        If you add the key `loss`, the reporter will report `main/loss` and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss` and `validation/main/loss` values.

        Returns:
            list: List of strings which are base keys to plot during training.

        """
        plot_keys = ['loss', 'l1_loss', 'mse_loss', 'bce_loss']
        if self.use_guided_attn_loss:
            plot_keys += ['attn_loss']
        if self.use_cbhg:
            plot_keys += ['cbhg_l1_loss', 'cbhg_mse_loss']
        return plot_keys
