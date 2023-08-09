import numpy as np
import pandas as pd
from termcolor import colored
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, cohen_kappa_score
import json
import wandb
import seaborn as sn
from matplotlib import pyplot as plt

def get_best_model(logdir, score_mode=None):
    best_score, best_step = 0., 0

    with open(logdir + '/stats.json', 'r') as f:
        log = json.load(f)

    for i, step in enumerate(log):
        current_score = float(log[step]['valid-' + score_mode])
        if best_score < current_score:
            best_score = current_score
            best_step = step

    print(f'best valid score [{score_mode}]:{best_score}')
    print(f'best valid step [{score_mode}]:{best_step}')

    net_dir = logdir + f'/_net_{best_step}.pth'

    return net_dir, best_step

def init_accumulator(engine):
    engine.accumulator = {'prob': [], 'pred': [], 'true': [], 'acc': [],'time': []}

def accumulate_outputs(engine):
    batch_output = engine.state.output
    for key, item in batch_output.items():
        if key not in ['loss', 'loss_main', 'loss_sub', 'margin']:
            engine.accumulator[key].extend([item])
    return

def update_log(output, epoch, prefix, color, log_file, commit=False):
    max_length = len(max(output.keys(), key=len))
    # print
    for metric in output:
        key = colored(prefix + '-' + metric.ljust(max_length) + ':', color)
        if metric in ['acc', 'f1_score', 'recall', 'precision', 'kappa', 'loss', 'loss_main', 'loss_sub', 'lr']:
            print("-----%s" % key, end=' ')
            print("%0.7f" % output[metric])
            if prefix in ['train', 'valid']:
                wandb.log({prefix + '-' + metric.ljust(max_length): output[metric]}, commit=False)
            else:
                wandb.log({f"{prefix}-{metric}": float(output[metric])}, commit=False)
        elif metric in ['conf_mat']:
            print("-----%s" % key, end=' ')
            conf_mat = output['conf_mat']
            conf_mat_df = pd.DataFrame(conf_mat)
            conf_mat_df.index.name = 'True'
            conf_mat_df.columns.name = 'Pred'
            output['conf_mat'] = conf_mat_df
            print('\n', conf_mat_df)
            plt.figure(figsize=(10, 7))
            sn.heatmap(conf_mat_df, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g')  # font size
            if prefix in ['train', 'valid']:
                wandb.log({prefix + '-' + "Conf_Mat": [wandb.Image(plt, caption="Label")]}, commit=commit)
            else:
                wandb.log({f"{prefix}-Conf_Mat": [wandb.Image(plt, caption="Label")]}, commit=False)
                wandb.log({f"{prefix}-EPOCH": epoch}, commit=commit)
            plt.title(f'{prefix} - Conf_Mat')
            plt.close()
        else:
            continue

    if prefix in ['train', 'valid']:
        # create stat dicts
        stat_dict ={}
        for metric in output:
            if metric in ['acc', 'f1_score', 'recall', 'precision', 'kappa', 'loss', 'loss_main', 'loss_sub']:
                metric_value = output[metric]
            elif metric in ['conf_mat']:
                conf_mat_df = pd.Series({'conf_mat': conf_mat}).to_json(orient='records')
                metric_value = conf_mat_df
            else:
                continue
            stat_dict['%s-%s' % (prefix, metric)] = metric_value

        # json stat log file, update and overwrite
        with open(log_file) as json_file:
            json_data = json.load(json_file)

        current_epoch = str(epoch)
        if current_epoch in json_data:
            old_stat_dict = json_data[current_epoch]
            stat_dict.update(old_stat_dict)
        current_epoch_dict = {current_epoch : stat_dict}
        json_data.update(current_epoch_dict)

        with open(log_file, 'w') as json_file:
            json.dump(json_data, json_file)

def process_accumulated_output(prefix, output, accumulator, batch_size, nr_classes, loss=None, loss_main=None, loss_sub=None):
    def uneven_seq_to_np(seq, size):
        item_count = size * (len(seq) - 1) + len(seq[-1])
        cat_array = np.zeros((item_count,) + seq[0][0].shape, seq[0].dtype)
        # BUG: odd len even
        for idx in range(0, len(seq) - 1):
            cat_array[idx * size: (idx + 1) * size] = seq[idx]
        cat_array[(idx + 1) * size:] = seq[-1]
        return cat_array

    if prefix == 'train':
        pred = uneven_seq_to_np(accumulator['pred'], batch_size)
        true = uneven_seq_to_np(accumulator['true'], batch_size)
    else:
        pred = uneven_seq_to_np(output['pred'], batch_size)
        true = uneven_seq_to_np(output['true'], batch_size)

    # threshold then get accuracy
    acc = np.mean(pred == true)

    # make confusion matrix
    precision = precision_score(true, pred, average='macro')
    recall = recall_score(true, pred, average='macro')
    f1 = f1_score(true, pred, average='macro')
    kappa = cohen_kappa_score(true, pred, weights='quadratic')
    conf_mat = confusion_matrix(true, pred, labels=np.arange(nr_classes))

    if prefix == 'train':
        proc_output = dict(acc=output['acc'],
                           precision=precision,
                           recall=recall,
                           f1_score=f1,
                           kappa=kappa,
                           loss=output['loss'],
                           loss_main=output['loss_main'],
                           loss_sub=output['loss_sub'],
                           lr=output['lr'],
                           conf_mat = conf_mat)
    elif prefix == 'valid':
        proc_output = dict(acc=acc,
                           precision=precision,
                           recall=recall,
                           f1_score=f1,
                           kappa=kappa,
                           loss=loss,
                           loss_main=loss_main,
                           loss_sub=loss_sub,
                           conf_mat=conf_mat,
                           )
    else:
        proc_output = dict(acc=acc,
                           precision=precision,
                           recall=recall,
                           f1_score=f1,
                           kappa=kappa,
                           loss=0,
                           loss_main=0,
                           loss_sub=0,
                           conf_mat=conf_mat,
                           )

    return proc_output

def training(engine, info):
    """
    running training measurement
    """
    train_ema_output = engine.state.metrics
    train_ema_output['lr'] = info['optimizer'].param_groups[0]['lr']

    output = process_accumulated_output(
        prefix='train',
        output=train_ema_output,
        accumulator=engine.accumulator,
        loss=None,
        loss_main=None,
        loss_sub=None,
        batch_size=info['train_batch_size'],
        nr_classes=info['nr_classes'])

    update_log(output, engine.state.epoch, 'train', 'green', info['json_file'])

def inference(engine, inferer, prefix, dataloader, info, commit):
    """
    running inference measurement
    """
    inferer.accumulator = {metric: [] for metric in info['metric_names']}
    inferer.run(dataloader)

    metrics = inferer.state.metrics
    output = process_accumulated_output(
        prefix=prefix,
        output=inferer.accumulator,
        accumulator=None,
        loss=metrics['loss'],
        loss_main=metrics['loss_main'],
        loss_sub=metrics['loss_sub'],
        batch_size=info['infer_batch_size'],
        nr_classes=info['nr_classes'])

    update_log(output, engine.state.epoch, prefix, 'red', info['json_file'], commit)

def testing(engine, prefix, info, commit=False):
    """
    running testing measurement
    """
    prefix = f"Best-{prefix}"

    output = process_accumulated_output(
        prefix=prefix,
        output=engine.accumulator,
        accumulator=None,
        batch_size=info['batch_size'],
        nr_classes=info['nr_classes'])

    update_log(output, info['best_step'], prefix, 'blue', None, commit)