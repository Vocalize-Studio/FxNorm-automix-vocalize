"""Config file."""

import numpy as np
import torch.nn as nn
from multiprocessing import cpu_count
from collections import OrderedDict

from automix.common_datatypes import DataType
from automix.common_losses import Loss, StereoLoss, StereoLoss
from automix.common_audioeffects import AugmentationChain

config = {}

""" GENERAL SETTINGS """

# Are we in debug mode?
config['DEBUG'] = False
config['CALCULATE_STATISTICS'] = True

config['OUTPUTS'] = ['vocals_normalized',
                    'bass_normalized',
                    'drums_normalized', 
                    'other_normalized']

config['INPUTS'] = ['vocals_normalized',
                    'bass_normalized',
                    'drums_normalized', 
                    'other_normalized'] #  Must follow names of normalized wav files. e.g.vocals_normalized.wa

# list of all sources (used for creating mixture)
config['SOURCES'] = config['INPUTS'] + config['OUTPUTS']
# list of all sources that should be learned
# (subset of `config['SOURCES']`; use tuples with several sources to learn joint models)
config['TARGETS'] = [('mixture',)]

# list of source mappings (to put several sources into one source group)
config['MAPPED_SOURCES'] = {}

# Save some information to visualize on tensorboard
config['TENSORBOARD'] = True


""" SETTINGS RELATED TO SIGNAL PROCESSING """

# Number of channels in the input data (1 = mono, 2 = stereo)
config['N_CHANNELS'] = 2

# Accepted sampling rates (list)
config['ACCEPTED_SAMPLING_RATES'] = [44100]

# FFT size (used only for loss computation here)
config['FFT_SIZE'] = 4096

config['HOP_LENGTH'] = config['FFT_SIZE'] // 4

# Analysis window for STFT (used only for loss computation here)
config['STFT_WINDOW'] = np.sqrt(np.hanning(config['FFT_SIZE']+1)[:-1])

# Number of bins of an STFT frame (used only for loss computation here)
config['N_BINS'] = config['FFT_SIZE'] // 2 + 1


""" SETTINGS RELATED TO DATA AUGMENTATION """

# Probability of source being present in the mixture
# (if not specified for a source then it is `1.` for that source)
config['PRESENT_PROBABILITY'] = {}

# Probability of source overlap, i.e., of having superposition of two stems
# for the same source. This can, e.g., be used to train a dialogue extraction
# with two speakers being active at the same time from single speaker data.
# (if not specified for a source then it is `0.` for that source)
config['OVERLAP_PROBABILITY'] = {}

# Initialize data augmentation chain
# Please see `common_audioeffects.py` for all available effects that can be used.
# In case you do not want to use any augmentation, just use `AugmentationChain()`.
                                       
config['AUGMENTER_CHAIN'] = AugmentationChain()
# In order to avoid any boundary effects, it is possible to input longer sequences into
# the augmenter and use a center-crop at its output, which should not be distorted by any boundary effects.
# Tuple contains the number of time samples that are added to the left/right.
config['AUGMENTER_SOURCES'] = []

config['AUGMENTER_PADDING'] = (0, 0)
config['SHUFFLE_STEMS'] = False
config['SHUFFLE_CHANNELS'] = True

""" SETTINGS RELATED TO NETWORK """

# Import network definition file
from automix.common_networkbuilding_cafx_tdcn_lstm_mix import Net, compute_receptive_field  # noqa E402, F401

# THIS PARAMETER MAY BE USELESS HERE, BUT REQUIRED IN train.py FOR NOW
config['NET_TYPE'] = 'CAFX_TDCN'

config['PRETRAIN'] = True
config['PRETRAIN_FRONT_END'] = False
# Initialization heuristic for network weights/biases
# (either `None` for PyTorch default or one of the initialization heuristics of `Net`)
config['INIT_NETWORK'] = None

# Number of features after the encoder (N)
config['N_FEATURES_ENCODER'] = 128

# Length of the encoder's filters, corresponding stride is half of it (L)
config['KERNEL_SIZE_ENCODER'] = 64

# Length of max pooling
config['MAX_POOLING'] = 64

# Number of features in the separation module, after bottleneck layer (B)
config['N_FEATURES_SEPARATION_MODULE'] = 256

# Number of features at the output of the skip connection path in a temporal block (Sc)
config['N_FEATURES_OUT'] = 64

# Number of features in the temporal blocks (H)
config['N_FEATURES_TB'] = 128

# Kernel size in convolutional blocks (P)
config['KERNEL_SIZE_TB'] = 3

# Number of temporal blocks in each repeat (X)
# Each repeat uses 2**0, 2**1, 2**2, 2**3, ... dilation factors in the 1D convolutions
config['N_TB_PER_REPEAT'] = 6

# Number of repeat (R)
config['N_REPEATS'] = 4

# SE amp ratio
config['SE_AMP_RATIO'] = 16

# Set how many samples we should discard in model output datatype
# to avoid boundary effects - e.g., due to receptive field of network
# (this is used for batched validation if `config['BATCHED_VALID'] = True` and
# can also be used during training if `guard_left`/`guard_right` are set for `config['TRAIN_LOSSES']`)

RF, guard = compute_receptive_field(config['KERNEL_SIZE_ENCODER'],
                                        config['KERNEL_SIZE_TB'],
                                        config['N_TB_PER_REPEAT'], 
                                        config['N_REPEATS'], config['MAX_POOLING'])

config['GUARD_LEFT'] = guard

config['GUARD_RIGHT'] = guard


""" SETTINGS RELATED TO TRAINING """

# Use cnDNN to benchmark convolution algorithms and selects the fastest,
# i.e. set `torch.backends.cudnn.benchmark = True`
config['CUDNN_BENCHMARK'] = True

# Number of sequences in each batch
config['BATCH_SIZE'] = 20

# Optimization learning rate
# (`LEARNING_RATES` is a list of tuples `(epochs, learning rate)`)
config['INITIAL_LEARNING_RATE'] = 1e-3
config['LEARNING_RATES'] = [(40, config['INITIAL_LEARNING_RATE']),
                            (20, config['INITIAL_LEARNING_RATE'] / 3.0),
                            (20, config['INITIAL_LEARNING_RATE'] / 10.0),
                            (10, config['INITIAL_LEARNING_RATE'] / 30.0),
                            (10, config['INITIAL_LEARNING_RATE'] / 100.0),
                            (5, config['INITIAL_LEARNING_RATE'] / 1000.0)]

# Define additional save points (i.e., epochs) where we store the network weights
config['SAVE_NET_AT_EPOCHS'] = []

# Add warmup phase for Adam/Amsgrad
config['LEARNING_RATES'].insert(0, (1, 0.0))
config['SAVE_NET_AT_EPOCHS'] = [_+1 for _ in config['SAVE_NET_AT_EPOCHS']]

# Compute total number of epochs
config['NUM_EPOCHS'] = np.sum(np.sum([_[0] for _ in config['LEARNING_RATES']]))

# Length of one training sequence (in network input datatype format)

config['TRAINING_SEQ_LENGTH'] = (3*min(config['ACCEPTED_SAMPLING_RATES']))//config['KERNEL_SIZE_ENCODER']
config['TRAINING_SEQ_LENGTH'] = config['TRAINING_SEQ_LENGTH']*config['KERNEL_SIZE_ENCODER']

# Frequency bins that we keep for processing (only up to 16khz to avoid instabilities)
# THIS IS NOT USEFUL FOR CONV-TASNET.
config['MAX_PROCESSED_FREQUENCY'] = np.minimum(16000.0, np.min(config['ACCEPTED_SAMPLING_RATES']) / 2.0)
config['N_BINS_KEEP'] = int(config['MAX_PROCESSED_FREQUENCY'] / (np.min(config['ACCEPTED_SAMPLING_RATES']) / 2.0) * config['N_BINS'])

# L2-Regularization (use `None` to fully switch it off)
config['L2_REGULARIZATION'] = 1e-6

# Gradient clipping
config['GRAD_CLIP_NORM_TYPE'] = 2  # e.g., `2` or `float('inf')
config['GRAD_CLIP_MAX_NORM'] = 0.2  # e.g., `0.1` or `None` (for adaptive clipping)

# Number of data providers that fill the queues
config['NUM_DATAPROVIDING_PROCESSES'] = cpu_count() // 2

# Number of minibatches that we use to estimate the input scale and offset as well as the warm-up for ADAM
config['NUM_MINIBATCHES_FOR_STATISTICS_ESTIMATION'] = 1000

# Number of minibatches that define one epoch, i.e., after which we compute the validation loss/store model snapshots
config['NUM_MINIBATCHES_PER_EPOCH'] = 1600


# Use AMSGrad optimizer
config['AMSGRAD'] = True

# Use quantization, if no quantization should be used then set to `None`
config['QUANTIZATION_OP'] = None
config['QUANTIZATION_BW'] = None

# Use mixed-precision support
config['USE_AMP'] = True

# Instead of processing full tracks for validation, use shorter segments having
# the same length as during training, stacked on the batch dimension in an overlap fashion
config['BATCHED_VALID'] = True


# Loss function(s) for training (`list`)

f_guard = int(np.ceil(((config['GUARD_LEFT'] - (config['FFT_SIZE'] - 1) - 1) / config['HOP_LENGTH']) + 1) + 1)


loss_1 = Loss(nn.L1Loss(), DataType.TIME_SAMPLES,
                            guard_left=config['GUARD_LEFT'],
                            guard_right=config['GUARD_RIGHT'])


config['TRAIN_LOSSES'] = [loss_1]

# Loss function(s) for validation (`OrderedDict`)
config['VALID_LOSSES'] = OrderedDict()
config['VALID_LOSSES']['td_l1'] = Loss(nn.L1Loss(), DataType.TIME_SAMPLES)


# Maximum sequence length that is used during validation -- currently choosen to be 5 minutes
# (we only use these many samples from the start of the WAV file)
config['MAX_VALIDATION_SEQ_LENGTH_TD'] = 4 * 60 * np.max(config['ACCEPTED_SAMPLING_RATES'])
# TODO: Increase if we have larger GPUs

# Specify folders where the training data is stored
config['DATA_DIR_TRAIN'] = []
config['DATA_DIR_TRAIN'].append(('/data/martinez/audio/automix/MUSDB18/train', False))

# Specify folders where the validation data is stored
config['DATA_DIR_VALID'] = []
config['DATA_DIR_VALID'].append(('/data/martinez/audio/automix/MUSDB18/val', False))

# The stems in the MUSDB18 are expected to be named vocals_normalized.wav, bass_normalized.wav, drums_normalized.wav, other_normalized.wav and mixture.wav
