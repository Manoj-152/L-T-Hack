import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import os
import augment
import resnet
import copy
import json
import utils
import wandb
import torch.nn.functional as F
import argparse
from datetime import datetime


MODELS = {
    "resnet18": resnet.resnet18,
    "resnet34": resnet.resnet34,
    "resnet50": resnet.resnet50,
    "resnet101": resnet.resnet101,
}


class Trainer:
    def __init__(self, args):
        self.args = args

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using device(s) = {self.device}")

        train_transform = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
        ]
        if args.autoaug:
            train_transform.append(augment.Policy(policy=args.autoaug_policy))
        train_transform.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        train_transform = transforms.Compose(train_transform)
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_sampler, val_sampler = None, None
        if args.dset == "crack":
            train_dset = datasets.ImageFolder(
                os.path.join(args.data_root, "train"), 
                transform=train_transform,
            )
            val_dset = datasets.ImageFolder(
                os.path.join(args.data_root, "valid"),
                transform=val_transform
            )
            num_cls = 2
        elif args.dset == "ps2":
            train_dset = datasets.ImageFolder(
                os.path.join(args.data_root, "train"), 
                transform=train_transform,
            )
            val_dset = datasets.ImageFolder(
                os.path.join(args.data_root, "test"),
                transform=val_transform
            )
            num_cls = 3
            img, target = train_dset[0]
            # print(img.shape, target)
            # exit()
        elif args.dset == "ship":
            dset = datasets.ImageFolder(
                os.path.join(args.data_root, "train"), 
                transform=train_transform,
            )
            train_dset, val_dset = dset, dset
            train_indx, val_indx = train_test_split(
                np.arange(len(dset)),
                test_size=0.1,
                shuffle=True,
                stratify=dset.targets,
            )
            train_sampler = SubsetRandomSampler(train_indx)
            val_sampler = SubsetRandomSampler(val_indx)
            num_cls = 5
        else:
            raise NotImplementedError(f"args.dset = {args.dset} is not implemented!")
        self.train_loader = DataLoader(
            train_dset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False if train_sampler is not None else True,
            sampler=train_sampler,
        )
        self.val_loader = DataLoader(
            val_dset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            sampler=val_sampler,
        )
        if train_sampler:
            print(f"number of samples, train: {len(train_indx)}, val: {len(val_indx)}")
        else:
            print(f"number of samples, train: {len(train_dset)}, val: {len(val_dset)}")

        assert args.net in MODELS.keys(), NotImplementedError(f"args.net = {args.net} is not implemented!")
        model = MODELS[args.net](num_cls=num_cls)
        teacher = copy.deepcopy(model)

        self.model = model.to(self.device)
        self.teacher = teacher.to(self.device)
        print(f"# of parameters: {sum(p.numel() for p in self.model.parameters())/1e6}M")

        self.ce_criterion = torch.nn.CrossEntropyLoss()
        self.kl_criterion = torch.nn.KLDivLoss()
        if args.optim == "sgd":
            self.optim = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        elif args.optim == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=1e-3)
        else:
            raise NotImplementedError(f"args.optim={args.optim} is not implemented")
        
        if os.path.exists(os.path.join(args.out_dir, "last.ckpt")):
            if not args.resume:
                raise ValueError(
                    f"args.out_dir={args.out_dir} exists, use --resume or change args.out_dir"
                )
            if self.args.use_best:
                ckpt = torch.load(os.path.join(args.out_dir, "best.ckpt"), map_location=self.device)
                self.model.load_state_dict(ckpt)
                self.train_epoch = 0
            else:
                ckpt = torch.load(os.path.join(args.out_dir, "last.ckpt"), map_location=self.device)
                self.model.load_state_dict(ckpt["model"])
                self.teacher.load_state_dict(ckpt["teacher"])
                self.optim.load_state_dict(ckpt["optim"])
                self.train_epoch = ckpt["train_epoch"] + 1
            print(f"loading ckpt from {args.out_dir}")
        else:
            if args.resume:
                raise ValueError(f"args.resume={args.resume}, no ckpt in {args.out_dir}")
            os.makedirs(args.out_dir, exist_ok=True)
            with open(os.path.join(args.out_dir, "args.txt"), "w") as f:
                json.dump(args.__dict__, f, indent=4)
            self.train_epoch = 0
            print(f"no ckpts from in {args.out_dir}")
        
        self.train_step = self.train_epoch * len(self.train_loader)
        self.log_wandb = False
        self.metric_meter = utils.AvgMeter()
        self.log_f = open(os.path.join(args.out_dir, "logs.txt"), "w")
        if args.wandb:
            self.log_wandb = True
            run = wandb.init()
            self.log_f.write(f"\nwandb url @ {run.get_url()}\n")

    
    def train_one_epoch(self):
        self.metric_meter.reset()
        self.model.train()
        for indx, (
            img,
            target,
        ) in enumerate(self.train_loader):
            metrics = {}

            img = img.to(self.device)
            target = target.to(self.device)

            feat, out = self.model(img, return_feat=True)
            loss_ce = self.ce_criterion(out, target)

            loss = loss_ce
            if self.args.use_teacher:
                t_feat, t_out = self.teacher(img, return_feat=True)
                loss_diff = (t_feat - feat).pow(2).mean()
                loss = loss_ce + loss_diff
                if self.args.distill_logits:
                    loss_kl = self.kl_criterion(F.log_softmax(out/self.args.temp, dim=1), F.softmax(t_out/self.args.temp, dim=1)) * (self.args.temp * self.args.temp)
                    loss = loss_ce * (1 - self.args.alpha) + loss_diff + loss_kl * self.args.alpha

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if self.args.use_teacher:
                for t_param, s_param in zip(self.teacher.parameters(), self.model.parameters()):
                    t_param.data.copy_(t_param.data * self.args.m + s_param.data * (1-self.args.m))

            pred_cls = out.argmax(dim=1)
            acc = pred_cls.eq(target.view_as(pred_cls)).sum().item() / img.shape[0]

            metrics["train_loss_ce"] = loss_ce.item()
            if self.args.use_teacher:
                metrics["train_loss_diff"] = loss_diff.item()
                # metrics["train_loss_kl"] = loss_kl.item()
            metrics["train_acc"] = acc
            self.metric_meter.add(metrics)
            utils.pbar(indx / len(self.train_loader), msg=self.metric_meter.msg())
            if self.log_wandb:
                wandb.log({"train_step": self.train_step, **metrics})
            self.train_step += 1
        utils.pbar(1, msg=self.metric_meter.msg())
    
    @torch.no_grad()
    def evaluate(self):
        self.metric_meter.reset()
        self.model.eval()
        for indx, (
            img,
            target,
        ) in enumerate(self.val_loader):
            metrics = {}

            img = img.to(self.device)
            target = target.to(self.device)

            feat, out = self.model(img, return_feat=True)
            loss_ce = self.ce_criterion(out, target)

            loss = loss_ce
            if self.args.use_teacher:
                t_feat, t_out = self.teacher(img, return_feat=True)
                loss_diff = (t_feat - feat).pow(2).mean()
                loss = loss_ce + loss_diff
                if self.args.distill_logits:
                    loss_kl = self.kl_criterion(F.log_softmax(out/self.args.temp, dim=1), F.softmax(t_out/self.args.temp, dim=1)) * (self.args.temp * self.args.temp)
                    loss = loss_ce * (1 - self.args.alpha) + loss_diff + loss_kl * self.args.alpha

            pred_cls = out.argmax(dim=1)
            acc = pred_cls.eq(target.view_as(pred_cls)).sum().item() / img.shape[0]

            metrics["val_loss_ce"] = loss_ce.item()
            if self.args.use_teacher:
                metrics["val_loss_diff"] = loss_diff.item()
                # metrics["val_loss_kl"] = loss_kl.item()
            metrics["val_acc"] = acc
            self.metric_meter.add(metrics)
            utils.pbar(indx / len(self.val_loader), msg=self.metric_meter.msg())
        utils.pbar(1, msg=self.metric_meter.msg())
    

    @torch.no_grad()
    def test(self):
        self.metric_meter.reset()
        self.model.eval()
        y_pred, y_true = [], []
        for indx, (
            img,
            target,
        ) in enumerate(self.val_loader):
            metrics = {}

            img = img.to(self.device)
            target = target.to(self.device)

            feat, out = self.model(img, return_feat=True)
            loss_ce = self.ce_criterion(out, target)

            loss = loss_ce
            if self.args.use_teacher:
                t_feat, t_out = self.teacher(img, return_feat=True)
                loss_diff = (t_feat - feat).pow(2).mean()
                loss = loss_ce + loss_diff
                if self.args.distill_logits:
                    loss_kl = self.kl_criterion(F.log_softmax(out/self.args.temp, dim=1), F.softmax(t_out/self.args.temp, dim=1)) * (self.args.temp * self.args.temp)
                    loss = loss_ce * (1 - self.args.alpha) + loss_diff + loss_kl * self.args.alpha

            pred_cls = out.argmax(dim=1)
            y_pred.extend(pred_cls.detach().cpu().numpy())
            y_true.extend(target.detach().cpu().numpy())
            acc = pred_cls.eq(target.view_as(pred_cls)).sum().item() / img.shape[0]

            metrics["val_loss_ce"] = loss_ce.item()
            if self.args.use_teacher:
                metrics["val_loss_diff"] = loss_diff.item()
                # metrics["val_loss_kl"] = loss_kl.item()
            metrics["val_acc"] = acc
            self.metric_meter.add(metrics)
            utils.pbar(indx / len(self.val_loader), msg=self.metric_meter.msg())
        utils.pbar(1, msg=self.metric_meter.msg())
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
        prec = precision_score(y_true, y_pred, average="micro")
        rec = recall_score(y_true, y_pred, average="micro")
        f1_score = f1_score(y_true, y_pred, average="micro")
        kappa_score = cohen_kappa_score(y_true, y_pred)
        print("metrics")
        print("-------------------------")
        print(f"precision: {prec}")
        print(f"recall: {rec}")
        print(f"f1_score: {f1_score}")
        print(f"kappa_score: {kappa_score}")

    def train(self):
        best_val = 0
        for self.train_epoch in range(self.train_epoch, self.args.train_epochs):
            print(f"epoch: {self.train_epoch}")
            print("-------------------------")
            self.train_one_epoch()
            self.log_f.write(f"[train_epoch]: {self.train_epoch} {self.metric_meter.msg()}\n")
            self.log_f.flush()

            self.evaluate()
            val_acc = self.metric_meter.get()["val_acc"]
            if val_acc > best_val:
                torch.save(self.model.state_dict(), os.path.join(self.args.out_dir, "best.ckpt"))
                print(f"val acc improved from {round(best_val, 4)} to {round(val_acc, 4)}")
                best_val = val_acc
            if self.log_wandb:
                wandb.log({"eval_epoch": self.train_epoch, **self.metric_meter.get()})

            self.log_f.write(
                f"[eval_epoch]: {self.train_epoch} {self.metric_meter.msg()}\n"
            )
            self.log_f.flush()

            torch.save(
                {
                    "model": self.model.state_dict(),
                    "teacher": self.teacher.state_dict() if self.args.use_teacher else None,
                    "optim": self.optim.state_dict(),
                    "train_epoch": self.train_epoch,
                },
                os.path.join(self.args.out_dir, "last.ckpt"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="set experiment seed [d: 42]")
    parser.add_argument("--use_teacher", action="store_true", help="use teacher [d: false]")
    parser.add_argument("--use_best", action="store_true", help="use best ckpt [d: false]")
    parser.add_argument("--distill_logits", action="store_true", help="distill logits [d: false]")
    parser.add_argument("--autoaug", action="store_true", help="use auto augment [d: false]")
    parser.add_argument(
        "--dset", type=str, default="crack", help="dataset name [d: crack]"
    )
    parser.add_argument("--data_root", type=str, required=True, help="dataset directory")
    parser.add_argument("--net", type=str, default="resnet50", help="network type [d: resnet50]")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size [d: 16]")
    parser.add_argument("--autoaug_policy", type=int, default=1, help="autoaug policy [d: 1]")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of workers for dataloading [d: 4]"
    )
    parser.add_argument("--alpha", type=float, default=0.4, help="alpha [d: 0.4]")
    parser.add_argument("--temp", type=float, default=6, help="temperature [d: 6]")
    parser.add_argument("--m", type=float, default=0.999, help="momentum in ema update [d: 0.999]")
    parser.add_argument("--optim", type=str, default="adam", help="optimizer name [d: adam]")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate [d: 1e-4]")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=f"out/{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
        help="path to output directory [d: out/year-month-date_hour-minute]",
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume training from checkpoint [d: false]"
    )
    parser.add_argument("--wandb", action="store_true", help="wandb logging [d: false]")
    parser.add_argument("--test", action="store_true", help="test method [d: false]")
    parser.add_argument("--train_epochs", type=int, default=300, help="training epochs [d: 300]")
    args = parser.parse_args()
    trainer = Trainer(args)
    if args.test:
        if args.dset == "crack":
            val_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            val_dset = datasets.ImageFolder(
                os.path.join(args.data_root, "test"),
                transform=val_transform
            )
            trainer.val_loader = DataLoader(
                val_dset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                sampler=None,
            )
        trainer.test()
    else:
        trainer.train()


