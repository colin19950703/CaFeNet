import os
import torch.nn
import argparse
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import random
import imgaug as ia

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.metrics import RunningAverage
from imgaug import augmenters as iaa
from utils.logger import *
from utils import check_log_dir
from model_lib.model_factory import create_model
import dataprepare
from config import CONFIGURE

torch.autograd.set_detect_anomaly(True)

class Trainer(CONFIGURE):
    def __init__(self, _args=None):
        super(Trainer, self).__init__(_args=_args)

    def check_manual_seed(self, seed):
        """
        If manual seed is not specified, choose a random one and notify it to the user
        """
        seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        ia.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        print('Using manual seed: {seed}'.format(seed=seed))
        return

    ####
    def eval_step(self, net, batch, device, sub_criterion=None):
        net.eval()  # infer mode

        imgs, true = batch
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        # push data to GPUs and convert to float32
        imgs = imgs.to(device).float()
        true = true.to(device).long()  # not one-hot

        # -----------------------------------------------------------
        with torch.no_grad():  # dont compute gradient
            loss = 0.
            loss_sub = torch.zeros(1).to(device)

            # assign output
            logit_class, features = net(imgs)

            # compute loss function
            loss_entropy = F.cross_entropy(logit_class, true, reduction='mean')
            loss += loss_entropy

            if sub_criterion is not None:
               _ = sub_criterion(features, true, margin=0, init_center=True)

            prob = F.softmax(logit_class, dim=-1)
            pred = torch.argmax(prob, dim=-1)
            # -----------------------------------------------------------
            return dict(
                loss=loss.item(),
                loss_main=loss_entropy.item(),
                loss_sub=loss_sub.item(),
                pred=pred.cpu().detach().numpy(),
                true=true.cpu().detach().numpy(),
            )

    def eval(self, score_mode, commit=False):
        if self.pretrained_weight == False:
            self.pretrained_dir, self.best_step = get_best_model(logdir=self.log_dir, score_mode=score_mode)

        # --------------------------- Dataloader
        _, valid_pairs, test_pairs = dataprepare.prepare_colon_data(data_root_dir=self.data_dir)

        infer_augmentors = self.infer_augmentors()

        valid_dataset = dataprepare.DatasetSerial(
            valid_pairs[:],
            shape_augs=iaa.Sequential(infer_augmentors[0]))
        test_dataset = dataprepare.DatasetSerial(
            test_pairs[:],
            shape_augs=iaa.Sequential(infer_augmentors[0]))

        valid_loader = data.DataLoader(
            valid_dataset,
            num_workers=self.nr_procs_valid,
            batch_size=self.infer_batch_size,
            shuffle=False,
            drop_last=False)
        test_loader = data.DataLoader(
            test_dataset,
            num_workers=self.nr_procs_valid,
            batch_size=self.infer_batch_size,
            shuffle=False,
            drop_last=False)

        # Define your network here
        net = create_model(model_name=self.model_name, num_classes=self.nr_classes, pretrained=self.pretrained_dir)
        net = torch.nn.DataParallel(net).to(self.device)

        log_info_dict = {
            'commit': commit,
            'score_mode': score_mode,
            'best_step': self.best_step,
            'nr_classes': self.nr_classes,
            'batch_size': self.infer_batch_size,
            'log_dir': self.log_dir
        }

        # --------------------------- Training Sequence
        valider = Engine(lambda engine, batch: self.eval_step(net, batch, self.device))
        tester = Engine(lambda engine, batch: self.eval_step(net, batch, self.device))

        # TODO: refactor this
        # attach running average metrics computation
        # decay of EMA to 0.95 to match tensorpack default
        timer = Timer(average=True)
        timer.attach(valider, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                     pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
        timer.attach(tester, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                     pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

        # attach progress bar
        pbar = ProgressBar(persist=True)
        pbar.attach(valider)
        pbar.attach(tester)


        valider.accumulator = {metric: [] for metric in ['pred', 'true']}
        tester.accumulator = {metric: [] for metric in ['pred', 'true']}

        valider.add_event_handler(Events.ITERATION_COMPLETED, accumulate_outputs)
        tester.add_event_handler(Events.ITERATION_COMPLETED, accumulate_outputs)
        valider.add_event_handler(Events.EPOCH_COMPLETED, testing, 'valid', log_info_dict, False)
        tester.add_event_handler(Events.EPOCH_COMPLETED, testing, 'test', log_info_dict, False)

        if self.data_name.lower() == 'colon':
            test2_pairs = dataprepare.prepare_colon_test2_data(data_root_dir=self.data_dir2)

            test2_dataset = dataprepare.DatasetSerial(
                test2_pairs[:],
                shape_augs=iaa.Sequential(infer_augmentors[0]))

            test2_loader = data.DataLoader(
                test2_dataset,
                num_workers=self.nr_procs_valid,
                batch_size=self.infer_batch_size,
                shuffle=False,
                drop_last=False)
            tester2 = Engine(lambda engine, batch: self.eval_step(net, batch, self.device))
            timer.attach(tester2, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                         pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
            pbar.attach(tester2)
            tester2.accumulator = {metric: [] for metric in ['pred', 'true']}
            tester2.add_event_handler(Events.ITERATION_COMPLETED, accumulate_outputs)
            tester2.add_event_handler(Events.EPOCH_COMPLETED, testing, 'test2', log_info_dict, log_info_dict['commit'])

            valider.run(valid_loader, 1)
            tester.run(test_loader, 1)
            tester2.run(test2_loader, 1)
        else:
            valider.run(valid_loader, 1)
            tester.run(test_loader, 1)
        return

    ####
    def run(self):
        modelname = f"{self.data_name}_{self.nr_classes}cls_{self.model_name}"
        print(modelname)

        wandb.login(key=self.wandb_key)
        wandb.init(
            project=f"MICCAI2023",  # _NIS or fft # , self.num_epochs
            entity=self.wandb_id,
            job_type="train",
            name=modelname,
            config=vars(args)
        )

        self.load_log_dir = self.log_dir + modelname
        self.log_dir = self.log_dir + modelname
        self.check_manual_seed(self.seed)
        self.eval('f1_score', commit=True)

        wandb.finish
        return


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    torch.backends.cudnn.enabled = False  # cuDNN error: CUDNN_status_mapping_error
    parser = argparse.ArgumentParser()
    ##
    parser.add_argument('--gpu', default='0,1', help='commaseparated list of GPU(s) to use.')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=5, help='number')
    #data
    parser.add_argument('--data_name', type=str, default='colon', choices=['colon'])
    parser.add_argument('--nr_classes', type=int, default=4, help='number')
    ##model
    parser.add_argument('--model_name', type=str, default='CaFeNet')
    parser.add_argument('--init_lr', type=float, default=1e-3, help='number')
    parser.add_argument('--margin', type=float, default=0.0, help='number')
    ##wandb
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--wandb_key', type=str, default=None)
    ##
    parser.add_argument('--pretrained_weight', action='store_true',
                        help='Set to True to use pretrained weights, or False to use weights from custom training.')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', args.device)
    print('Count of using GPUs:', torch.cuda.device_count())
    print('Current cuda device:', torch.cuda.current_device())

    trainer = Trainer(_args=args)
    trainer.run()
