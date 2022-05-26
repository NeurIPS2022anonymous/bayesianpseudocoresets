import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from torchvision.utils import save_image
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, match_loss
import copy
import random
from reparam_module import ReparamModule
from eval_hmc import evaluate_hmc
from eval_sghmc import evaluate_sghmc

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
from log import get_logger

def main(args):
    
    setting = args.bpc_method
    if args.bpc_method == 'fKL':
        setting += '_lrinit_' + str(args.lr_init)
    setting += '_expert_' + str(args.expert_epochs)
    setting += '_noise_' + str(args.noise_scale_student)
    setting += '_lrimg_' + str(args.lr_img)
    if args.no_aug :
        setting += '_noaug'
        args.buffer_path = 'buffers_no_aug'
    
    workdir = os.path.join('bpc_final_%s_ipc%d'%(args.dataset, args.ipc), setting)
    # workdir = 'trash'

    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    logfilename = os.path.join(workdir, time.strftime('%d-%I-%M-%S_') + '%s.log'%str(setting))
    logger = get_logger(logfilename)

    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        logger.info("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    logger.info("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean1, std1, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1


    logger.info('Hyper-parameters: \n %s' %args.__dict__)
    logger.info('Evaluation model pool: %s' %model_eval_pool)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    logger.info("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        logger.info('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        logger.info('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    def get_real_batch(n):
        idx_shuffle = np.random.permutation(len(labels_all))[:n]
        images = images_all[idx_shuffle]
        labels = labels_all[idx_shuffle]
        return images, labels

    ''' initialize the synthetic data '''
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    if args.texture:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
    else:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)

    syn_lr = torch.tensor(args.lr_init).to(args.device)

    if args.pix_init == 'real':
        logger.info('initialize synthetic data from random real images')
        if args.texture:
            for c in range(num_classes):
                for i in range(args.canvas_size):
                    for j in range(args.canvas_size):
                        image_syn.data[c * args.ipc:(c + 1) * args.ipc, :, i * im_size[0]:(i + 1) * im_size[0],
                        j * im_size[1]:(j + 1) * im_size[1]] = torch.cat(
                            [get_images(c, 1).detach().data for s in range(args.ipc)])
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    else:
        logger.info('initialize synthetic data from random noise')


    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)

    scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=args.Iteration, eta_min=0)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img.zero_grad()

    criterion = nn.CrossEntropyLoss().to(args.device)
    logger.info('%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset)#, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    logger.info("Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        logger.info("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}
    
    for it in range(0, args.Iteration+1):
        save_this_it = False

        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                logger.info('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    logger.info('DSA augmentation strategy: \n %s' %args.dsa_strategy)
                    logger.info('DSA augmentation parameters: \n %s' %args.dsa_param.__dict__)
                else:
                    logger.info('DC augmentation parameters: \n %s' %args.dc_aug_param)

                accs_test = []
                accs_train = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                    # net_eval2 = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

                    eval_labs = label_syn
                    with torch.no_grad():
                        image_save = image_syn
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification

                    args.lr_net = syn_lr.item()
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)
                    logger.info('----map acc test %.4f'%acc_test)
                    # _, loss_test, acc_test, ece, conf = evaluate_sghmc(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, logger)
                    # _, acc_train, acc_test = evaluate_synset_ensemble(it_eval, net_eval, net_eval2, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)
                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)
                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                    best_acc_it = it
                logger.info('-------------------------\nIter %d Evaluate %d random %s, mean = %.4f std = %.4f'%(it, len(accs_test), model_eval, acc_test_mean, acc_test_std))
                logger.info('Best it %d acc %.4f\n-------------------------'%(best_acc_it, best_acc[model_eval]))
                # print(loss_test)

            with torch.no_grad():
                image_save = image_syn.cuda()

                save_dir = os.path.join(workdir, 'pic')
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)

                torch.save(image_save.cpu(), os.path.join(workdir, "images_%d.pt"%(it)))
                torch.save(label_syn.cpu(), os.path.join(workdir, "labels_%d.pt"%(it)))

                # pic save # 
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std1[ch] + mean1[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, os.path.join(save_dir, 'images_%d.png'%(it)), nrow=args.ipc)

                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(workdir, "images_best.pt"))
                    torch.save(label_syn.cpu(), os.path.join(workdir, "labels_best.pt"))


        ### training ####
        if args.bpc_method == 'W' or args.bpc_method == 'fKL' or args.bpc_method == 'rKL':
            student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model

            student_net = ReparamModule(student_net)

            if args.distributed:
                student_net = torch.nn.DataParallel(student_net)

            student_net.train()

            num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

            if args.load_all:
                expert_trajectory = buffer[np.random.randint(0, len(buffer))]
            else:
                expert_trajectory = buffer[expert_idx]
                expert_idx += 1
                if expert_idx == len(buffer):
                    expert_idx = 0
                    file_idx += 1
                    if file_idx == len(expert_files):
                        file_idx = 0
                        random.shuffle(expert_files)
                    # logger.info("loading file {}".format(expert_files[file_idx]))
                    if args.max_files != 1:
                        del buffer
                        buffer = torch.load(expert_files[file_idx])
                    if args.max_experts is not None:
                        buffer = buffer[:args.max_experts]
                    random.shuffle(buffer)

            start_epoch = np.random.randint(0, args.max_start_epoch)
            starting_params = expert_trajectory[start_epoch]
            target_params = expert_trajectory[start_epoch+args.expert_epochs]

            target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)
            student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
            starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)


            syn_images = image_syn

            y_hat = label_syn.to(args.device)

            param_loss_list = []
            param_dist_list = []
            indices_chunks = []

            for step in range(args.syn_steps):
                if not indices_chunks:
                    indices = torch.randperm(len(syn_images))
                    indices_chunks = list(torch.split(indices, args.batch_syn))

                these_indices = indices_chunks.pop()

                x = syn_images[these_indices]
                this_y = y_hat[these_indices]

                if args.texture:
                    x = torch.cat([torch.stack([torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)), torch.randint(im_size[1]*args.canvas_size, (1,))), (1,2))[:,:im_size[0],:im_size[1]] for im in x]) for _ in range(args.canvas_samples)])
                    this_y = torch.cat([this_y for _ in range(args.canvas_samples)])

                if args.dsa and (not args.no_aug):
                    x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

                if args.distributed:
                    forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
                else:
                    forward_params = student_params[-1]
                x = student_net(x, flat_param=forward_params)
                ce_loss = criterion(x, this_y)

                if args.bpc_method == 'W':
                    grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
                else:
                    grad = torch.autograd.grad(ce_loss, student_params[-1])[0]

                student_params.append(student_params[-1] - syn_lr * grad)

            if args.bpc_method == 'W':   
                param_loss = torch.tensor(0.0).to(args.device)
                param_dist = torch.tensor(0.0).to(args.device)

                param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
                param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

                param_loss_list.append(param_loss)
                param_dist_list.append(param_dist)


                param_loss /= num_params
                param_dist /= num_params

                param_loss /= param_dist

                grand_loss = param_loss

                optimizer_img.zero_grad()
                optimizer_lr.zero_grad()

                grand_loss.backward()

                optimizer_img.step()
                optimizer_lr.step()

                for _ in student_params:
                    del _

                if it%10 == 0:
                    logger.info('tm %s iter = %04d, tm_loss = %.4f' %(get_time(), it, grand_loss.item()))
                    logger.info('syn_lr: %s'%(str(syn_lr.item())))
            elif args.bpc_method == 'fKL': 
                # pm_loss = torch.tensor(0.0).to(args.device)
                optimizer_img.zero_grad()
                for num in range(args.num_pm_samples):
                    if args.dsa and (not args.no_aug):
                        x = DiffAugment(image_syn, args.dsa_strategy, param=args.dsa_param)
                    else:
                        x = image_syn
                    current_param = student_params[-1].clone().detach()  ## unroll gradient detach ## 
                    noise1 = torch.randn_like(student_params[-1]) * args.noise_scale_student
                    ce_loss_student = criterion(student_net(x, flat_param = current_param+noise1), label_syn)
                    
                    noise2 = torch.randn_like(student_params[-1]) * args.noise_scale_student
                    ce_loss_teacher = criterion(student_net(x, flat_param = target_params+noise2), label_syn)
                    pm_loss = (- ce_loss_student + ce_loss_teacher)/args.num_pm_samples
                    pm_loss.backward()
                
                optimizer_img.step()
                # scheduler_img.step()

                for _ in student_params:
                    del _

                if it%10 == 0:
                    logger.info('pm single %s iter = %04d, pm_loss = %.4f' %(get_time(), it, pm_loss.item()))
            
            else: 
                gs = torch.zeros(args.num_pm_samples, args.batch_real)
                gs_tilde = torch.zeros(args.num_pm_samples, args.ipc*num_classes)
                h_tilde = torch.zeros((args.num_pm_samples,) + image_syn.size())
                criterion_bpsvi = nn.CrossEntropyLoss(reduction='none').to(args.device)

                real_images, real_labels = get_real_batch(args.batch_real)
                real_images, real_labels = real_images.cuda(), real_labels.cuda()  # TODO: aug

                if args.dsa and (not args.no_aug):
                    real_images = DiffAugment(real_images, args.dsa_strategy, param=args.dsa_param)
                for num in range(args.num_pm_samples):
                    # unroll gradient detach ##
                    current_param = student_params[-1].clone().detach()
                    noise1 = torch.randn_like(student_params[-1]) * args.noise_scale_student

                    output_real = student_net(
                        real_images, flat_param=current_param + noise1)
                    gs[num] = - criterion_bpsvi(output_real, real_labels)

                    output_syn = student_net(
                        image_syn, flat_param=current_param + noise1)
                    gs_tilde[num] = - criterion_bpsvi(output_syn, label_syn)

                    _images = image_syn.clone().data.requires_grad_()
                    if args.dsa and (not args.no_aug):
                        _images_aug = DiffAugment(_images, args.dsa_strategy, param=args.dsa_param)
                    else:
                        _images_aug = _images
                    outputs = student_net(_images_aug, flat_param=current_param + noise1)
                    potential = -criterion_bpsvi(outputs, label_syn)
                    grad = torch.autograd.grad(
                        potential, _images, grad_outputs=torch.ones_like(potential))[0]
                    h_tilde[num] = grad.clone().detach()

                torch.set_grad_enabled(False)

                gs = gs - gs.mean(0)
                gs_tilde = gs_tilde - gs_tilde.mean(0)
                h_tilde = h_tilde - h_tilde.mean(0)

                image_grad = torch.zeros(image_syn.size())
                for s in range(args.num_pm_samples):
                    image_grad += h_tilde[s] * (gs[s].mean() - gs_tilde[s].mean())
                image_grad /= args.num_pm_samples
                image_syn = image_syn + args.lr_img * image_grad.cuda()

                torch.set_grad_enabled(True)

                for _ in student_params:
                    del _

                if it % 10 == 0:
                    logger.info('%s iter = %04d' % (get_time(), it))
        
        else:
            print('check bpc type!')
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=1, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-5, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.03, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=1, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=30, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=40, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")
    parser.add_argument('--distributed', action='store_true')

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    parser.add_argument('--bpc_method', type=str, default='fKL', help='W, rKL, fKL')
    
    parser.add_argument('--num_pm_samples', type=int, default=30, help='samples for calculating expected potential')
    parser.add_argument('--noise_scale_student', type=float, default=0.01, help='pm noise scale')
    parser.add_argument('--noise_scale_teacher', type=float, default=0.01, help='pm noise scale')

    ## samling hps ##
    parser.add_argument('--wd', type=float, default=1.5, help='hmc weight decay')
    parser.add_argument('--eps', type=float, default=1e-2, help='hmc lr')
    parser.add_argument('--theta_scale', type=float, default=0.1, help='hmc initial theta scale')
    parser.add_argument('--mom_scale', type=float, default=0.1, help='hmc initial momentum scale')
    parser.add_argument('--num_iter', type=int, default=100, help='hmc num samples')
    parser.add_argument('--num_train', type=int, default=100, help='')
    parser.add_argument('--alpha', type=float, default=0.1, help='')
    parser.add_argument('--T', type=float, default=0.01, help='')
    parser.add_argument('--num_lf', type=int, default=5, help='hmc leapfrog steps')
    parser.add_argument('--burn', type=int, default=50, help='hmc burnin')
    parser.add_argument('--thin', type=int, default=1, help='hmc thin')

    
    args = parser.parse_args()

    main(args)


