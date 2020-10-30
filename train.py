import torch
import torch.optim as optim
import torch.nn as nns
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import data_utils
import numpy as np
from tqdm import trange
import torch.nn as nn

parser = argparse.ArgumentParser()
# 학습률 = 0.0002
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
# Adam Optimizer 의 파라미터 Beta1 = 0.9
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
# 배치사이즈 = 1
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
# 로그 파일을 저장할 폴더
parser.add_argument('--log_dir', default='logs', help='base directory to save logs')
# 모델 파일을 저장할 폴더
parser.add_argument('--model_dir', default='', help='base directory to save logs')
# 학습 모델 이름 설정 (이 이름으로 폴더 생성)
parser.add_argument('--name', default='', help='identifier for directory')
# 데이터 파일 저장된 폴더
parser.add_argument('--data_root', default='smmnist', help='root directory for data')
# Optimizer 종류 설정 = adam
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
# Epoch 설정 = 100
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
# 랜덤 Seed
parser.add_argument('--seed', default=1, type=int, help='manual seed')
# Epoch 당 반복 횟수 = 2000
parser.add_argument('--epoch_size', type=int, default=2000, help='epoch size')
# 이미지 너비, 높이, 채널
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--image_height', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=8, type=int)
# 데이터셋 이름
parser.add_argument('--dataset', default='smmnist', help='dataset to train with')
# 입력데이터에서 과거 프레임의 수
parser.add_argument('--n_past', type=int, default=10, help='number of frames to condition on')
# 출력 프레임의 미래 프레임 수
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
# 평가할 때 실제 예측 할 프레임의 수
parser.add_argument('--n_eval', type=int, default=10, help='number of frames to predict at eval time')
# RNN 은닉층의 차원 = 512
parser.add_argument('--rnn_size', type=int, default=128, help='dimensionality of hidden layer')
# 후기, 예측 RNN 레이어의 수
parser.add_argument('--posterior_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=6, help='number of layers')
# 프레임 인터벌 = 1
parser.add_argument('--gap', type=int, default=1, help='number of timesteps')
# Z 잠재공간의 차원
parser.add_argument('--z_dim', type=int, default=128, help='dimensionality of z_t')
# 인코더의 출력 / 디코드의 입력 차원
parser.add_argument('--g_dim', type=int, default=128,
                    help='dimensionality of encoder output vector and decoder input vector')
# KL(쿨백라이블러) 가중치 = 0.0001
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
# 데이터 로딩 쓰레드(훈련할 땐 안쓰임)
parser.add_argument('--data_threads', type=int, default=0, help='number of data loading threads')

parser.add_argument('--num_digits', type=int, default=2, help='number of digits')

opt = parser.parse_args()

# 모델 불러오기
if opt.model_dir != '':
    # 불러올 모델이 있는 경우
    saved_model = torch.load('%s/model.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    # 기존 모델의 속성 가져오기
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir
else:
    # 불러올 모델을 지정하지 않은 경우
    name = 'model_city_trial=%dx%d-rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.7f-g_dim=%d-z_dim=%d-beta=%.7f%s' % (opt.image_width, opt.image_width, opt.rnn_size, opt.predictor_rnn_layers, opt.posterior_rnn_layers, opt.n_past, opt.n_future, opt.lr, opt.g_dim, opt.z_dim,  opt.beta, opt.name)
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)
# 생성된 결과 및 플롯을 저장할 폴더 생성
os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

# 입력 옵션을 바탕으로 학습에 필요한 변수들 생성 (랜덤변수생성)
opt.max_step = opt.n_past+opt.n_future
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor
opt.data_type = 'sequence'

# ---------------- 콘솔에 모델 설정 값 출력 ----------------

print(opt)

# ---------------- Optimizer ----------------
# 근데 왜 옵션으로 Optimizer 받아놓고 안꺼내쓰는지 모르겠음 ;;
opt.optimizer = optim.Adam

# 로컬 모듈 불러오기
import layers as model

# 예측 모델 생성 (지그재그 리버스 예측자)
frame_predictor = model.zig_rev_predictor(opt.rnn_size,
                                          opt.rnn_size,
                                          opt.g_dim,
                                          opt.predictor_rnn_layers,
                                          opt.batch_size,
                                          'lstm',
                                          int(opt.image_width/16),
                                          int(opt.image_height/16))

# 인코더 모델 생성
encoder = model.autoencoder(nBlocks=[2, 2, 2, 2], nStrides=[1, 2, 2, 2],
                            nChannels=None, init_ds=2,
                            dropout_rate=0., affineBN=True,
                            in_shape=[opt.channels, opt.image_width, opt.image_height], mult=4)


# 예측자와 인코더 각각 Optimizer 생성
frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# --------- 손실함수 ------------------------------------
mse_criterion = nn.MSELoss()

# --------- GPU에 모델 적재 ------------------------------------
frame_predictor.cuda()
encoder.cuda()
mse_criterion.cuda()

# --------- 데이터셋 불러오기 ------------------------------------
train_data, test_data = data_utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          #num_workers=2,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=False)
test_loader = DataLoader(test_data,
                         #num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=False,
                         drop_last=False,
                         pin_memory=False)


# 배치 사이즈만큼 데이터를 분할해서 하나씩 리턴
def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = data_utils.normalize_data(opt, dtype, sequence)
            yield batch


training_batch_generator = get_training_batch()


def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = data_utils.normalize_data(opt, dtype, sequence)
            yield batch


testing_batch_generator = get_testing_batch()


# 플롯 그리기
def plot(x, epoch, p=False):
    nsample = 1  # 샘플의 수
    gen_seq = [[] for i in range(nsample)]  # 생성된 시퀸스
    gt_seq = [x[i][:, :3] for i in range(len(x))]  # 그라운드 트루스
    mse = 0
    for s in range(nsample):
        # 초기 히든레이어 정의
        frame_predictor.hidden = frame_predictor.init_hidden()
        # 잠재공간 정의
        memo = Variable(torch.zeros(opt.batch_size, opt.rnn_size, int(opt.image_width/16), int(opt.image_height/16)).cuda())

        # 시퀸스 정의 (왜 3개만 가져옴???)
        gen_seq[s].append(x[0][:, :3])
        x_in = x[0]
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if i < opt.n_past:
                _, memo, _ = frame_predictor((h,memo))
                x_in = x[i]
                gen_seq[s].append(x_in[:,:3])
            else:
                h_pred, memo, _ = frame_predictor((h, memo))
                x_in = encoder(h_pred, False).detach()
                gen_seq[s].append(x_in[:, :3])

    to_plot = []
    gifs = [[] for t in range(opt.n_eval)]
    nrow = min(opt.batch_size, 10)

    for t in range(opt.n_past, opt.n_eval):
        for i in range(opt.batch_size):
            mse += torch.sum((gt_seq[t][i][:, :495, :436].data.cpu() - gen_seq[0][t][i][:, :495, :436].data.cpu()) ** 2) / (495.*436.*9.)

    for i in range(nrow):
        row = []
        for t in range(opt.n_eval):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        s_list = [0]
        for ss in range(len(s_list)):
            s = s_list[ss]
            row = []
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i])
            to_plot.append(row)
        for t in range(opt.n_eval):
            row = []
            row.append(gt_seq[t][i])
            for ss in range(len(s_list)):
                s = s_list[ss]
                row.append(gen_seq[s][t][i])
            gifs[t].append(row)

    if p:
        fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch)
        data_utils.save_tensors_image(fname, to_plot)

        fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch)
        data_utils.save_gif(fname, gifs)
    return mse


# --------- training functions ------------------------------------
def train(x, e):
    frame_predictor.zero_grad()
    encoder.zero_grad()
    frame_predictor.hidden = frame_predictor.init_hidden()

    mse = 0
    memo = Variable(torch.zeros(opt.batch_size, opt.rnn_size, int(opt.image_width/16), int(opt.image_height/16)).cuda())

    for i in range(1, opt.n_past + opt.n_future):
        h = encoder(x[i - 1], True)
        h_pred, memo, _ = frame_predictor((h,memo))
        x_pred = encoder(h_pred, False)
        mse += (mse_criterion(x_pred, x[i]))

    loss = mse
    loss.backward()

    frame_predictor_optimizer.step()
    encoder_optimizer.step()

    return mse.data.cpu().numpy() / (opt.n_past + opt.n_future)


# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    frame_predictor.train()
    encoder.train()
    epoch_mse = 0

    for i in trange(opt.epoch_size):
        x = next(training_batch_generator)
        mse = train(x, epoch)
        epoch_mse += mse

    with torch.no_grad():
        frame_predictor.eval()
        encoder.eval()

        eval = 0
        for i in range(360):
            x = next(testing_batch_generator)
            if i == 0:
                ssim = plot(x, epoch, True)
            else:
                ssim = plot(x, epoch)
            eval += ssim

        print('[%02d] mse loss: %.7f| ssim loss: %.7f(%d)' % (
            epoch, epoch_mse / opt.epoch_size, eval / 360.0, epoch * opt.epoch_size * opt.batch_size))

    # save the model
    if epoch % 1 == 0:
        torch.save({
            'encoder': encoder,
            'frame_predictor': frame_predictor,
            'opt': opt},
            '%s/model_%s.pth' % (opt.log_dir, epoch))
