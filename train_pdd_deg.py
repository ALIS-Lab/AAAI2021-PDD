import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import itertools

import progress
from torch.autograd import Variable
from networks.ensemble_resnet import ensemble_3_resnet18_fc, ensemble_5_resnet18_fc, ensemble_3_resnet34_fc, ensemble_5_resnet34_fc
from config.dataset_config import getData

from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser(description='PyTorch CNN Training')
parser.add_argument(
    '--model',
    type=str,
    default='ensemble_3_resnet18_fc',
    help='CNN architecture')
parser.add_argument('--dataset', type=str, default='CIFAR100', help='datasets') #Tiny_Image FashionMNIST CIFAR100
parser.add_argument('--bs', default=64, type=int, help='batch size')
parser.add_argument('--lr', default=0.02, type=float, help='learning rate')
parser.add_argument('--alpha', default=1.0, type=float, help='alpha')
parser.add_argument('--beta', default=0.01, type=float, help='beta')
parser.add_argument(
    '--resume',
    '-r',
    action='store_true',
    help='resume from checkpoint')
parser.add_argument(
    '--save_dir',
    type=str,
    default='ensemble_3_resnet18_fc',
    help='save log and model')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_Test_acc = 0
best_Test_acc_epoch = 0
start_epoch = 0 

if opt.dataset == 'FashionMNIST':
    total_epoch = 40
if opt.dataset == 'CIFAR100':
    total_epoch = 150
if opt.dataset == 'Tiny_Image':
    total_epoch = 100

path = os.path.join('./models', opt.dataset, opt.save_dir+'_'+opt.model)
if not os.path.isdir(path):
    os.mkdir(path)
results_log_csv_name = opt.save_dir + '_results.csv'

# Data
print('==> Preparing data..')

# setup data loader
num_classes, train_data, test_data = getData(opt.dataset)
trainloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=opt.bs,
    shuffle=True,
    num_workers=4,
    pin_memory=True)
testloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=40, #tiny:20
    shuffle=False,
    num_workers=4,
    pin_memory=True)

# Model
if opt.model == 'ensemble_3_resnet18_fc':
    num_models = 3
    net = ensemble_3_resnet18_fc(num_classes, num_models)
elif opt.model == 'ensemble_5_resnet18_fc':
    num_models = 5
    net = ensemble_5_resnet18_fc(num_classes, num_models)
elif opt.model == 'ensemble_3_resnet34_fc':
    num_models = 3
    net = ensemble_3_resnet34_fc(num_classes, num_models)
elif opt.model == 'ensemble_5_resnet34_fc':
    num_models = 5
    net = ensemble_5_resnet34_fc(num_classes, num_models)
else:
    raise NotImplementedError

# setup optimizer
optimizer = optim.SGD(
    net.parameters(),
    lr=opt.lr,
    momentum=0.9,
    weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_epoch, eta_min=1e-4)

# setup checkpoint
if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path, 'best_model.pth'))

    net.load_state_dict(checkpoint['net'])
    best_Test_acc = checkpoint['acc']
    best_Test_acc_epoch = checkpoint['epoch']
    start_epoch = checkpoint['epoch'] + 1
    for x in range(start_epoch):
        scheduler.step()
else:
    print('==> Preparing %s %s %d' %(opt.model, opt.dataset, total_epoch))
    print('==> Building model..')

# covert net to GPU
if use_cuda:
    net = net.cuda()

class LogNLLLoss(nn.Module):
    
    def __init__(self, reduction='mean', off_set=1e-64):
        super(LogNLLLoss, self).__init__()
        assert reduction == 'mean'
        self.reduction = reduction
        self.offset = off_set

    def forward(self, x, targets):
        log_x = torch.log(x+self.offset)
        log_nll_loss = nn.NLLLoss()(log_x, targets)
        return log_nll_loss


class GradPenalty(nn.Module):

    def __init__(self):
        super(GradPenalty, self).__init__()

    def forward(self, x, y_list):
        assert type(y_list).__name__ == 'list' and len(y_list) > 0
        grads = []

        for y_i in y_list:
            grad_i, = torch.autograd.grad(y_i, x, retain_graph=True, create_graph=True)
            grads.append(grad_i)
        gp_loss = self.grad_diversity_penalty(grads)

        return gp_loss

    def grad_diversity_penalty(self, grads):
        zero_div_off = 1e-64
        nums = len(grads)
        idx_list = list(range(nums))
        cos_grads_angles = 0
        for idx_tup in itertools.combinations(idx_list, 2):
            grad_a = grads[idx_tup[0]]
            grad_b = grads[idx_tup[1]]
            grad_dot = torch.sum(torch.mul(grad_a, grad_b), dim=[1,2,3])
            grad_a_norm = torch.sqrt(torch.sum(grad_a**2, dim=[1,2,3]))
            grad_b_norm = torch.sqrt(torch.sum(grad_b**2, dim=[1,2,3]))
            grad_norm_prod = grad_a_norm * grad_b_norm + zero_div_off
            cos_grad_angle = grad_dot / grad_norm_prod
            cos_grads_angles += torch.mean(cos_grad_angle)
        return cos_grads_angles


def main():

    # record train log
    with open(os.path.join(path, results_log_csv_name), 'w') as f:
        f.write('epoch, train_loss, test_loss, train_ja_loss, test_ja_loss, train_acc, test_acc, time\n')

    # start train
    for epoch in range(start_epoch, total_epoch):
        print('current time:', datetime.now().strftime('%b%d-%H:%M:%S'))
        train(epoch)
        test(epoch)
        # Log results
        with open(os.path.join(path, results_log_csv_name), 'a') as f:
            f.write('%5d, %.5f, %.5f, %.5f, %.5f, %.6f, %.5f, %s,\n' %( epoch,
                                                                  train_loss,
                                                                  Test_loss,
                                                                  train_gp_loss,
                                                                  Test_gp_loss,
                                                                  Train_acc,
                                                                  Test_acc,
                                                                  datetime.now().strftime('%b%d-%H:%M:%S')))

    print("best_Test_acc: %.3f" % best_Test_acc)
    print("best_Test_acc_epoch: %d" % best_Test_acc_epoch)

    # best ACC
    with open(os.path.join(path, results_log_csv_name), 'a') as f:
        f.write('%s,%03d,%0.3f,\n' %('best acc (test)',
                                      best_Test_acc_epoch,
                                      best_Test_acc))


def ensemble_CE(logits, target, num_models=3):
    assert logits.shape[-1] % num_models == 0
    num_splits = int(logits.shape[-1] / num_models)
    criterion = LogNLLLoss()
    logits_split = torch.split(logits, num_splits, dim=-1)
    ce_loss = 0
    losses = []
    for logit in logits_split:
        ce_loss_i = criterion(logit, target) / num_models
        losses.append(ce_loss_i)
        ce_loss += ce_loss_i
    return ce_loss, losses

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    global train_loss
    global train_gp_loss
    net.train()
    train_loss = 0
    train_gp_loss = 0
    correct = 0
    total = 0

    scheduler.step()
    print('learning_rate: %s' % str(scheduler.get_lr()))
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
        outputs = net(inputs)

        CE_loss, CE_loss_list = ensemble_CE(outputs, targets, num_models)
        GP_loss = GradPenalty()(inputs, CE_loss_list)
        loss = opt.alpha*CE_loss + opt.beta*GP_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.data
        train_gp_loss += GP_loss.data

        output_split = torch.split(outputs, num_classes, dim=-1)
        for out in output_split:
            _, p = torch.max(out.data, 1)
            correct += p.eq(targets.data).cpu().sum() / num_models
        total += targets.size(0)

        progress.progress_bar(
            batch_idx,
            len(trainloader),
            'Total_Loss: %.3f GP_Loss: %.3f | Acc: %.3f%% (%d/%d)'%(train_loss /(batch_idx +1),
                                                                    train_gp_loss /(batch_idx +1),
                                                                    100. *float(correct) /total,
                                                                    correct,
                                                                    total))
    Train_acc = 100. * float(correct) / total


def test(epoch):
    global Test_acc
    global best_Test_acc
    global best_Test_acc_epoch
    global Test_loss
    global Test_gp_loss
    net.eval()
    Test_loss = 0
    Test_gp_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
        outputs = net(inputs)

        CE_loss, CE_loss_list = ensemble_CE(outputs, targets, num_models)
        GP_loss = GradPenalty()(inputs, CE_loss_list)
        loss = opt.alpha * CE_loss + opt.beta * GP_loss

        Test_loss += loss.data
        Test_gp_loss += GP_loss.data

        output_split = torch.split(outputs, num_classes, dim=-1)
        for out in output_split:
            _, p = torch.max(out.data, 1)
            correct += p.eq(targets.data).cpu().sum() / num_models
        total += targets.size(0)

        progress.progress_bar(
            batch_idx,
            len(testloader),
            'Total_Loss: %.3f GP_Loss: %.3f | Acc: %.3f%% (%d/%d)' %(Test_loss /(batch_idx + 1),
                                                                     Test_gp_loss/(batch_idx + 1),
                                                                     100. *float(correct)/total,
                                                                     correct,
                                                                     total))

    # Save checkpoint.
    Test_acc = 100. * float(correct) / total
    if Test_acc > best_Test_acc:
        print('Saving..')
        print("best_Test_acc: %0.3f" % Test_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'acc': Test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'best_model.pth'))
        best_Test_acc = Test_acc
        best_Test_acc_epoch = epoch


if __name__ == '__main__':
    main()