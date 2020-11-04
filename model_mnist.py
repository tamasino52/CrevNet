# Library import
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import time
from skimage.measure import compare_ssim
from tqdm import trange
import easydict

# Local Module
import pssim.pytorch_ssim as pytorch_ssim
import utils.data_utils as data_utils
import utils.utils as utils
import utils.utils_3d as utils_3d
from DOFLoss import DOFLoss

# Option parameter
opt = easydict.EasyDict({})
opt.lr = 0.0005
opt.beta1 = 0.9
opt.batch_size = 1
opt.log_dir = 'logs'
opt.model_dir = ''
opt.name = 'mymodel'
opt.data_root = 'data'
opt.optimizer = optim.Adam
opt.data_type = 'sequence'
opt.niter = 60
opt.epoch_size = 5000
opt.image_width = 64
opt.channels = 1
opt.dataset = 'smmnist'
opt.n_past = 8
opt.n_future = 10
opt.n_eval = 18
opt.rnn_size = 32
opt.predictor_rnn_layers = 8
opt.beta = 0.0001
opt.model = 'crevnet'
opt.data_threads = 0
opt.num_digits = 2
opt.max_step = opt.n_past + opt.n_future + 2

# Random seed setting
opt.seed = 1
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

# Folder naming
if opt.model_dir != '':
    saved_model = torch.load('%s/model.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir
else:
    name = 'model_custom=layers_%s=seq_len_%s=batch_size_%s' % (opt.predictor_rnn_layers,opt.n_eval,opt.batch_size)
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

# Model setting
from AutoEncoder.AutoEncoder import AutoEncoder
from RPM.ReversiblePredictor import ReversiblePredictor

frame_predictor = ReversiblePredictor(input_size=opt.rnn_size,
                                      hidden_size=opt.rnn_size,
                                      output_size=opt.rnn_size,
                                      n_layers=opt.predictor_rnn_layers,
                                      batch_size=opt.batch_size)

encoder = AutoEncoder(nBlocks=[4,5,3],
                      nStrides=[1, 2, 2],
                      nChannels=None,
                      init_ds=2,
                      dropout_rate=0.,
                      affineBN=True,
                      in_shape=[opt.channels, opt.image_width, opt.image_width],
                      mult=2)

frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

scheduler1 = torch.optim.lr_scheduler.StepLR(frame_predictor_optimizer, step_size=50, gamma=0.2)
scheduler2 = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=50, gamma=0.2)

mse_criterion = nn.MSELoss()

# Transfer to GPU
frame_predictor.cuda()
encoder.cuda()
mse_criterion.cuda()

# Dataset loading
train_data, test_data = data_utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=False)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=False)


# Batch generating
def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = data_utils.normalize_data(opt, dtype, sequence)
            yield batch


def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = data_utils.normalize_data(opt, dtype, sequence)
            yield batch


training_batch_generator = get_training_batch()
testing_batch_generator = get_testing_batch()


# Plot
def plot(x, epoch, p=False):
    nsample = 1
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]
    mse = 0
    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        memo = Variable(torch.zeros(opt.batch_size, opt.rnn_size, 3, int(opt.image_width/8), int(opt.image_width/8)).cuda())
        gen_seq[s].append(x[0])
        x_in = x[0]
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if i < opt.n_past:
                _,memo = frame_predictor((h,memo))
                x_in = x[i]
                gen_seq[s].append(x_in)
            elif i == opt.n_past:
                h_pred, memo = frame_predictor((h, memo))
                x_in = encoder(h_pred, False).detach()
                x_in[:, :, 0] = x[i][:, :, 0]
                x_in[:, :, 1] = x[i][:, :, 1]
                gen_seq[s].append(x_in)
            elif i == opt.n_past + 1:
                h_pred, memo = frame_predictor((h, memo))
                x_in = encoder(h_pred, False).detach()
                x_in[:, :, 0] = x[i][:, :, 0]
                gen_seq[s].append(x_in)
            else:
                h_pred, memo = frame_predictor((h, memo))
                x_in =encoder(h_pred,False).detach()
                gen_seq[s].append(x_in)

    to_plot = []
    gifs = [[] for t in range(opt.n_eval)]
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        # ground truth sequence
        row = []
        for t in range(opt.n_eval):
            row.append(gt_seq[t][i][0][2])
        to_plot.append(row)
        mse = 0
        for s in range(nsample):
            for t in range(opt.n_past,opt.n_eval):
                mse += pytorch_ssim.ssim(gt_seq[t][:,0,2][:,None, :, :],gen_seq[s][t][:,0,2][:,None, :, :])
        s_list = [0]
        for ss in range(len(s_list)):
            s = s_list[ss]
            row = []
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i][0][2])
            to_plot.append(row)
        for t in range(opt.n_eval):
            row = []
            row.append(gt_seq[t][i][0][2])
            for ss in range(len(s_list)):
                s = s_list[ss]
                row.append(gen_seq[s][t][i][0][2])
            gifs[t].append(row)

    if p:
        fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch)
        data_utils.save_tensors_image(fname, to_plot)

        fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch)
        data_utils.save_gif(fname, gifs)
    return mse / 10.0


# Train
def train(x, e, DOF_Loss=None):
    frame_predictor.zero_grad()
    encoder.zero_grad()

    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()
    MSE_Loss = 0
    DOF_Loss = 0
    memo = Variable(torch.zeros(opt.batch_size, opt.rnn_size, 3, int(opt.image_width/8), int(opt.image_width/8)).cuda())

    for i in range(1, opt.n_past + opt.n_future):
        h = encoder(x[i - 1], True)
        h_pred, memo = frame_predictor((h, memo))
        x_pred = encoder(h_pred, False)
        MSE_Loss += mse_criterion(x[i], x_pred)
        gen_diff, gt_diff = DOFLoss.dense_optical_flow_loss(x_pred, x[i], opt.channels)

        for i in range(len(gen_diff)):
            DOF_Loss += DOFLoss.calc_optical_flow_loss(gen_diff[i], gt_diff[i], torch.device('cuda:0')) / 2

    # 역전파 적용
    loss = 0.8 * MSE_Loss + 0.2 * DOF_Loss
    loss.backward()

    frame_predictor_optimizer.step()
    encoder_optimizer.step()

    return MSE_Loss.data.cpu().numpy() / (opt.n_past + opt.n_future)


# Train loop
for epoch in range(opt.niter):
    frame_predictor.train()
    encoder.train()
    epoch_mse = 0

    for i in trange(opt.epoch_size):
        x = next(training_batch_generator)
        input = []
        for j in range(opt.n_eval):
            k1 = x[j][:, 0][:, None, None, :, :]
            k2 = x[j + 1][:, 0][:, None, None, :, :]
            k3 = x[j + 2][:, 0][:, None, None, :, :]

            input.append(torch.cat((k1,k2,k3),2))
        mse = 0
        mse = train(input, epoch)
        epoch_mse += mse

    scheduler1.step()
    scheduler2.step()

    with torch.no_grad():
        frame_predictor.eval()
        encoder.eval()

        eval = 0
        for i in range(100):
            x = next(testing_batch_generator)
            input = []
            for j in range(opt.n_eval):
                k1 = x[j][:, 0][:, None, None, :, :]
                k2 = x[j + 1][:, 0][:, None, None, :, :]
                k3 = x[j + 2][:, 0][:, None, None, :, :]

                input.append(torch.cat((k1, k2, k3), 2))
            if i == 0:
                ssim = plot(input, epoch, True)
            else:
                ssim = plot(input, epoch)
            eval += ssim

        print('[%02d] mse loss: %.7f| ssim loss: %.7f(%d)' % (
            epoch, epoch_mse / opt.epoch_size,eval/ 100.0, epoch * opt.epoch_size * opt.batch_size))

    # save the model
    if epoch % 10 == 0:
        torch.save({
            'encoder': encoder,
            'frame_predictor': frame_predictor,
            'opt': opt},
            '%s/model_%s.pth' % (opt.log_dir, epoch))

    torch.save({
        'encoder': encoder,
        'frame_predictor': frame_predictor,
        'opt': opt},
        '%s/model.pth' % opt.log_dir)