import argparse
import logging
import os
import shutil
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Dataset.UnifiedDataset import UnifiedDataset
from ModelSimilarity import ModelSimilarity
from Utils import FocalLoss, make_weights_for_balanced_classes, SMLossAVT
from SAM import SAM

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Seperates: -, =
def parse_args():
    parser = argparse.ArgumentParser()

    # Names, paths, logs
    parser.add_argument("--ckpt_path", default="./ckpt")
    parser.add_argument("--log_path", default="./log")
    parser.add_argument("--task_name", default="test")

    # Data parameters
    parser.add_argument("--a_dim", default=74, type=int)
    parser.add_argument("--v_dim", default=35, type=int)
    parser.add_argument("--t_dim", default=300, type=int)
    parser.add_argument("--data_type", default='MMMO', type=str)
    parser.add_argument("--task", default='classification', type=str) # classification, regression
    parser.add_argument("--normalize", default=0, type=int)
    parser.add_argument("--pin_memo", default=0, type=int)
    parser.add_argument("--workers_num", default=8, type=int)
    parser.add_argument("--num_class", default=7, type=int) # 1, 2, 3, 7, 17
    parser.add_argument("--droplast", default=0, type=int)
    
    # Model parameters
    parser.add_argument("--d_inner", default=512, type=int)
    parser.add_argument("--layers", default=2048, type=int)
    parser.add_argument("--n_head", default=8, type=int)
    parser.add_argument("--mask", default=1, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--d_out", default=128, type=int)
    parser.add_argument("--n_position", default=128, type=int)
    parser.add_argument("--feature_aug", default='mean', type=str)
    parser.add_argument("--feature_compose", default='sum', type=str)
    parser.add_argument("--add_sa", default=0, type=int)

    # Training and optimization
    parser.add_argument("--sm_loss", default='gau', type=str) # gau, kl
    parser.add_argument("--sm_level", default='vec', type=str) # batch, vec, time
    parser.add_argument("--gamma", default=1.0, type=float) # ele, vec, time

    parser.add_argument("--seed", default=0, type=int) # 1,0
    parser.add_argument("--weighted_sampler", default=1, type=int) # 1,0
    parser.add_argument("--loss", default='CE', type=str) # Focal, CE, MSE
    parser.add_argument("--epochs_num", default=70, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--optm", default="Adam", type=str)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--learning_rate", default=4e-3, type=float)
    parser.add_argument("--lr_decrease", default='step', type=str) # multi_step
    parser.add_argument("--lr_decrease_iter", default='60', type=str)
    parser.add_argument("--lr_decrease_rate", default=0.1, type=float)
    parser.add_argument("--cuda", default="0", type=str)
    parser.add_argument("--parallel", default=1, type=int)

    opt = parser.parse_args()

    opt.normalize = opt.normalize == 1
    opt.mask = opt.mask == 1
    opt.add_sa = opt.add_sa == 1
    opt.parallel = opt.parallel == 1
    opt.pin_memo = opt.pin_memo == 1
    opt.droplast = opt.droplast == 1
    opt.weighted_sampler = opt.weighted_sampler == 1

    if 'IEMOCAP' in opt.data_type:
        print('The metrics for IEMOCAP is currently not sure, therefore currently not supported.')
        exit()

    return opt


def make_dataset(opt):
    train_dataset = UnifiedDataset(mode="train", normalize=opt.normalize, type=opt.data_type)
    valid_dataset = UnifiedDataset(mode="valid", normalize=opt.normalize, type=opt.data_type)
    test_dataset = UnifiedDataset(mode="test", normalize=opt.normalize, type=opt.data_type)

    if opt.weighted_sampler and opt.task=='classification':
        weights = torch.DoubleTensor(make_weights_for_balanced_classes(train_dataset.label, opt.num_class))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle,
        batch_size=opt.batch_size,
        persistent_workers=True,
        num_workers=opt.workers_num,
        pin_memory=opt.pin_memo,
        drop_last=opt.droplast,
        sampler=sampler,
    )
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=opt.batch_size,
        persistent_workers=True,
        num_workers=opt.workers_num,
        pin_memory=opt.pin_memo,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=opt.batch_size,
        persistent_workers=True,
        num_workers=opt.workers_num,
        pin_memory=opt.pin_memo,
        drop_last=False,
    )
    return train_loader, valid_loader, test_loader


def get_model(opt):
    model = ModelSimilarity(d_a=opt.a_dim, d_v=opt.v_dim, d_t=opt.t_dim, 
        d_inner=opt.d_inner, layers=opt.layers, n_head=opt.n_head, 
        dropout=opt.dropout, d_out=opt.d_out, 
        feature_aug=opt.feature_aug, feature_compose=opt.feature_compose, add_sa=opt.add_sa,
        num_class=opt.num_class, n_position=opt.n_position)
    return model


def get_labels_from_datas(datas, opt):
    labels = None
    if opt.task == 'classification':
        if opt.data_type in ['Youtube', 'MOUD', 'MMMO', ]:
            labels = datas[3].cuda().long()
        if opt.data_type in ['MOSI_20', 'MOSI_50', 'MOSEI_20', 'MOSEI_50' ]:
            labels = datas[3][1].cuda().long() if opt.num_class==2 else datas[3][2].cuda().long()
        if opt.data_type in ['POM']:
            labels = datas[3][1].cuda().long()
    elif opt.task == 'regression':
        # if opt.data_type in ['POM']:
        #     labels = [data_.cuda().float() for data_ in datas[3]]
        if opt.data_type in ['POM', 'MOSI_20', 'MOSI_50', 'MOSEI_20', 'MOSEI_50' ]:
            labels = datas[3][0].cuda().float()
    else:
        raise NotImplementedError
    assert labels is not None, 'Labels are None due to the conflict between task and num_class'

    return labels

def get_outputs_from_datas(model, a_data, v_data, t_data, opt):
    outputs = model(a_data, v_data, t_data, 
        get_mask_from_sequence(a_data, -1) if opt.mask else None, 
        get_mask_from_sequence(v_data, -1) if opt.mask else None, 
        get_mask_from_sequence(t_data, -1) if opt.mask else None,
        return_features=True)

    return outputs


def get_loss_from_loss_func(outputs, labels, loss_func, opt):
    outputs, V_F, A_F, T_F = outputs
    loss_func, sm_loss_func = loss_func
    if opt.loss in ['Focal', 'CE']:
        loss = loss_func(outputs, labels)
    elif opt.loss in ['BCE']:
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=opt.num_class).float()
        loss = loss_func(outputs, labels_one_hot)
    elif opt.loss in ['RMSE', 'MAE']:
        if opt.data_type in ['MOSI_20', 'MOSI_50', 'MOSEI_20', 'MOSEI_50', 'POM' ]:
            loss = loss_func(outputs.reshape(-1, ), labels.reshape(-1, ))
        # elif opt.data_type in ['POM']:
        #     # out: [bs, 17], labels: [bs] * 17
        #     loss = loss_func(outputs, torch.stack(labels, -1))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    loss_sm = sm_loss_func((V_F, A_F, T_F))

    return (1-opt.gamma)*loss + opt.gamma*loss_sm 


def get_score_from_result(predictions_corr, labels_corr, opt):
    if opt.task == 'classification':
        _, predictions_corr = topk_(predictions_corr, 1, 1)
        predictions_corr, labels_corr = predictions_corr.reshape(-1,), labels_corr.reshape(-1,)
        acc = accuracy_score(labels_corr, predictions_corr)
        f1_micro = f1_score(labels_corr, predictions_corr, average='micro')
        f1_macro = f1_score(labels_corr, predictions_corr, average='macro')
        if opt.data_type == 'POM':
            mae = mean_absolute_error(predictions_corr, labels_corr)
            acc = (acc, mae, f1_macro)
        else:
            acc = (acc, f1_micro, f1_macro)
    elif opt.task == 'regression':
        acc = [mean_absolute_error(labels_corr, predictions_corr)]
    else :
        raise NotImplementedError

    return acc


def get_optimizer(opt, model):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if opt.optm == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=float(opt.learning_rate),
            weight_decay=opt.weight_decay,
        )
    elif opt.optm == "SGD":
        optimizer = torch.optim.SGD(
            params,
            lr=float(opt.learning_rate),
            weight_decay=opt.weight_decay,
            momentum=0.9
        )
    elif opt.optm == "SAM":
        optimizer = SAM(
            params,
            torch.optim.Adam,
            lr=float(opt.learning_rate),
            weight_decay=opt.weight_decay,
        )
    else:
        raise NotImplementedError

    if opt.lr_decrease == 'step':
        opt.lr_decrease_iter = int(opt.lr_decrease_iter)
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_decrease_iter, opt.lr_decrease_rate)
    elif opt.lr_decrease == 'multi_step':
        opt.lr_decrease_iter = list((map(int, opt.lr_decrease_iter.split('='))))
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_decrease_iter, opt.lr_decrease_rate)
    else:
        raise NotImplementedError
    return optimizer, lr_schedule


def rmse(output, target):
    output, target = output.reshape(-1, ), target.reshape(-1,)
    rmse_loss = torch.sqrt(((output - target) ** 2).mean())
    return rmse_loss


def get_loss(opt):
    if opt.loss == 'Focal':
        loss_func = FocalLoss()
    elif opt.loss == 'CE':
        loss_func = torch.nn.CrossEntropyLoss()
    elif opt.loss == 'BCE':
        loss_func = torch.nn.BCEWithLogitsLoss()
    elif opt.loss == 'RMSE':
        loss_func = rmse
    elif opt.loss == 'MAE':
        loss_func = torch.nn.L1Loss()
    
    return (loss_func, SMLossAVT(opt.sm_loss, opt.sm_level))


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def main():
    opt = parse_args()

    # Set random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.set_device("cuda:" + opt.cuda)

    # Set Logger
    os.makedirs(os.path.join(opt.ckpt_path, opt.task_name), exist_ok=True)
    os.makedirs(os.path.join(opt.log_path, opt.task_name, "predictions"), exist_ok=True)
    set_logger(os.path.join(opt.log_path, opt.task_name, "log.log"))
    writer = SummaryWriter(os.path.join(opt.log_path, opt.task_name))
    best_model_name_val = os.path.join(opt.ckpt_path, opt.task_name, "best_model_val.pth.tar")
    best_model_name_test = os.path.join(opt.ckpt_path, opt.task_name, "best_model_test.pth.tar")
    ckpt_model_name = os.path.join(opt.ckpt_path, opt.task_name, "latest_model.pth.tar")

    logging.log(msg="Making dataset and model", level=logging.DEBUG)

    # Get dataloader, loss, optimizer and model
    train_loader, valid_loader, test_loader = make_dataset(opt)
    model = get_model(opt)
    optimizer, lr_schedule = get_optimizer(opt, model)
    loss_func = get_loss(opt)

    if opt.parallel:
        logging.log(msg="Model paralleling", level=logging.DEBUG)
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    best_acc_val, best_acc_test = -1, -1
    best_val_predictions, best_test_predictions = None, None

    start_epoch = 0
    logging.log(msg="Start training", level=logging.DEBUG)
    for epoch in range(start_epoch, int(opt.epochs_num)):
        # Do Train and Evaluate
        train_loss, train_acc = train(train_loader, model, optimizer, loss_func, opt)
        val_loss, val_acc, val_predictions = evaluate(valid_loader, model, loss_func, opt)
        test_loss, test_acc, test_predictions = evaluate(test_loader, model, loss_func, opt)
        lr_schedule.step()

        # Save Model
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
        }
        torch.save(state, ckpt_model_name)

        # Updata metrics, results and features
        if val_acc[0] > best_acc_val:
            shutil.copyfile(ckpt_model_name, best_model_name_val)
            best_acc_val, best_val_predictions = val_acc[0], val_predictions
        if test_acc[0] > best_acc_test:
            shutil.copyfile(ckpt_model_name, best_model_name_test)
            best_acc_test, best_test_predictions = test_acc[0], test_predictions

        if opt.task == 'regression':
            msg = (
                "epoch: [{:3.0f}]".format(epoch + 1)
                + " train_loss: [{0:.5f}]".format(train_loss)
                + " train_score: [{0:6.3f}] ".format(train_acc[0]) 
                + " val_loss: [{0:.5f}]".format(val_loss)
                + " val_score: [{0:6.3f}] ".format(val_acc[0])
                + " test_loss: [{0:.5f}]".format(test_loss)
                + " test_score: [{0:6.3f}] ".format(test_acc[0]) 
            )
        else:
            msg = (
            "epoch: [{:3.0f}]".format(epoch + 1)
            + " train_loss: [{0:.5f}]".format(train_loss)
            + " train_score(a/fi/fa): [{0:6.3f}][{1:6.3f}][{2:6.3f}] ".format(train_acc[0], train_acc[1], train_acc[2])
            + " val_loss: [{0:.5f}]".format(val_loss)
            + " valid_score(a/fi/fa): [{0:6.3f}][{1:6.3f}][{2:6.3f}] ".format(val_acc[0], val_acc[1], val_acc[2])
            + " test_loss: [{0:.5f}]".format(test_loss)
            + " valid_score(a/fi/fa): [{0:6.3f}][{1:6.3f}][{2:6.3f}] ".format(test_acc[0], test_acc[1], test_acc[2])
        )
        logging.log(msg=msg, level=logging.DEBUG)
        
        writer.add_scalar('Train/Epoch/Loss', train_loss, epoch)
        writer.add_scalar('Train/Epoch/Acc', train_acc[0], epoch)
        writer.add_scalar('Valid/Epoch/Loss', val_loss, epoch)
        writer.add_scalar('Valid/Epoch/Acc', val_acc[0], epoch)
        writer.add_scalar('Test/Epoch/Loss', test_loss, epoch)
        writer.add_scalar('Test/Epoch/Acc', test_acc[0], epoch)
        writer.add_scalar('Lr',  lr_schedule.get_last_lr()[-1], epoch)

    logging.log(msg="Best Val score: {0:.5f}".format(best_acc_val), level=logging.DEBUG)
    logging.log(msg="Best Test score: {0:.5f}".format(best_acc_test), level=logging.DEBUG)
    writer.close()

    # Save predictions
    np.save(os.path.join(opt.log_path, opt.task_name, "predictions", "val.npy"), best_val_predictions)
    np.save(os.path.join(opt.log_path, opt.task_name, "predictions", "test.npy"), best_test_predictions)


def train(train_loader, model, optimizer, loss_func, opt):
    model.train()
    running_loss, predictions_corr, labels_corr = 0.0, [], []

    for _, datas in enumerate(train_loader):
        a_data, v_data, t_data = datas[0].cuda().float(), datas[1].cuda().float(), datas[2].cuda().float()
        labels = get_labels_from_datas(datas, opt)

        outputs = get_outputs_from_datas(model, a_data, v_data, t_data, opt)
        loss = get_loss_from_loss_func(outputs, labels, loss_func, opt)

        if opt.optm == 'SAM':
            loss.backward()
            optimizer.first_step(zero_grad=True)
            outputs = get_outputs_from_datas(model, a_data, v_data, t_data, opt)
            loss = get_loss_from_loss_func(outputs, labels, loss_func, opt)
            loss.backward()
            optimizer.second_step(zero_grad=True)
            running_loss += loss.item()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        with torch.no_grad():
            predictions_corr += outputs[0].cpu().numpy().tolist()
            labels_corr += labels.cpu().numpy().tolist() # if opt.data_type != 'POM' else torch.stack(labels, -1).cpu().numpy().tolist()

    predictions_corr, labels_corr = np.array(predictions_corr), np.array(labels_corr)
    train_acc = get_score_from_result(predictions_corr, labels_corr, opt)

    return running_loss/len(train_loader), train_acc


def evaluate(val_loader, model, loss_func, opt):
    model.eval()
    running_loss, predictions_corr, labels_corr = 0.0, [], []
    with torch.no_grad():
        for _, datas in enumerate(val_loader):
            a_data, v_data, t_data = datas[0].cuda().float(), datas[1].cuda().float(), datas[2].cuda().float()
            labels = get_labels_from_datas(datas, opt)
            outputs = get_outputs_from_datas(model, a_data, v_data, t_data, opt)
            loss = get_loss_from_loss_func(outputs, labels, loss_func, opt)

            running_loss += loss.item()
            predictions_corr += outputs[0].cpu().numpy().tolist()
            labels_corr += labels.cpu().numpy().tolist() # if opt.data_type != 'POM' else torch.stack(labels, -1).cpu().numpy().tolist()

    predictions_corr, labels_corr = np.array(predictions_corr), np.array(labels_corr)
    val_acc = get_score_from_result(predictions_corr, labels_corr, opt)

    return running_loss/len(val_loader), val_acc, predictions_corr


def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort, topk_index_sort


def get_seperate_acc(labels, predictions, num_class):
    accs = [0 for i in range(num_class)]
    alls = [0 for i in range(num_class)]
    corrects = [0 for i in range(num_class)]
    for label, prediction in zip(labels, predictions):
        alls[label] += 1
        if label == prediction:
            corrects[label] += 1
    for i in range(num_class):
        accs[i] = '{0:5.1f}%'.format(100 * corrects[i] / alls[i])
    return ','.join(accs)


def get_mask_from_sequence(sequence, dim):
    return (torch.sum(torch.abs(sequence), dim=dim) == 0)


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    main()
