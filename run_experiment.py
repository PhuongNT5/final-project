import os
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser

from model_arch import YOLO_Temporal_Transformer
from temporal_video_loader import UnsafeNetTemporalDataset

# 1. Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 2. Save experiment configuration
def save_config(config, output_dir):
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=4)

# 3. Plot and save confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, epoch, output_dir):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cm_epoch_{epoch}.png'))
    plt.close()

# 4. Main experiment function
def run_experiment(args=None):
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--freeze_until_epoch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--backbone_lr_mult', type=float, default=0.2, help='Backbone LR = lr * multiplier when unfrozen')
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--output_dir', type=str, default='experiment_results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_sampler', action='store_true', help='Use WeightedRandomSampler for class imbalance')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision (CUDA only)')
    parser.add_argument('--yolo_weights', type=str, default='yolo11s.pt')
    parser.add_argument('--feature_layer_idx', type=int, default=-3)
    parser.add_argument('--mode', type=str, default='video', choices=['video', 'features'])
    parser.add_argument('--train_features', type=str, default='')
    parser.add_argument('--test_features', type=str, default='')
    parser.add_argument('--best_metric', type=str, default='macro_f1', choices=['macro_f1', 'acc'])
    parser.add_argument('--head_dim', type=int, default=512, help='Transformer model dimension for features mode (input is projected to this dim).')
    parser.add_argument('--max_t', type=int, default=32, help='Max T supported by token head positional embeddings (features mode tokens).')
    parser.add_argument('--max_s', type=int, default=64, help='Max S supported by token head positional embeddings (features mode tokens).')
    parser.add_argument('--use_cls_token', action='store_true', help='Use CLS token pooling for tokenized features (recommended).')
    parser.add_argument('--focal_gamma', type=float, default=0.0, help='If >0, use focal loss with this gamma (features mode).')
    args = parser.parse_args(args)

    config = {
        'seed': args.seed,
        'epochs': args.epochs,
        'freeze_until_epoch': args.freeze_until_epoch,
        'learning_rate': args.lr,
        'backbone_lr_mult': args.backbone_lr_mult,
        'batch_size': args.batch_size,
        'sequence_length': args.sequence_length,
        'img_size': args.img_size,
        'output_dir': args.output_dir,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_sampler': bool(args.use_sampler),
        'amp': bool(args.amp) and torch.cuda.is_available(),
        'yolo_weights': args.yolo_weights,
        'feature_layer_idx': args.feature_layer_idx,
        'head_dim': int(args.head_dim),
        'max_t': int(args.max_t),
        'max_s': int(args.max_s),
        'use_cls_token': bool(args.use_cls_token),
        'focal_gamma': float(args.focal_gamma),
    }
    os.makedirs(config['output_dir'], exist_ok=True)
    set_seed(config['seed'])
    save_config(config, config['output_dir'])

    if args.mode == 'features':
        from feature_dataset import PreExtractedFeatureDataset
        from temporal_head import TemporalTransformerHead
        from spatiotemporal_head import SpatioTemporalTransformerHead
        from torch.utils.data import WeightedRandomSampler

        train_set = PreExtractedFeatureDataset(args.train_features)
        test_set = PreExtractedFeatureDataset(args.test_features)

        # class weights (fast)
        labels = np.array([int(train_set.labels[i].item()) for i in range(len(train_set))], dtype=int)
        num_classes = 8
        class_counts = np.bincount(labels, minlength=num_classes)
        class_counts_safe = np.maximum(class_counts, 1)
        inv = labels.size / (num_classes * class_counts_safe)
        class_weights = (inv / inv.mean()).astype(np.float32)
        config['class_weights'] = class_weights.tolist()

        # sampler to balance batches (recommended)
        sample_w = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(sample_w),
            replacement=True,
        )

        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=False, sampler=sampler, num_workers=int(args.num_workers))
        test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=int(args.num_workers))

        x0, _ = train_set[0]
        if x0.dim() == 2:
            # (T,C) => vector features
            in_dim = int(x0.shape[-1])
            model = TemporalTransformerHead(
                in_dim=in_dim,
                model_dim=int(config['head_dim']),
                num_classes=8,
                nhead=8,
                num_layers=4,
                dropout=0.5,
            ).to(config['device'])
        elif x0.dim() == 3:
            # (T,S,C) => token features
            in_dim = int(x0.shape[-1])
            model = SpatioTemporalTransformerHead(
                in_dim=in_dim,
                model_dim=int(config['head_dim']),
                num_classes=8,
                nhead=8,
                num_layers=4,
                dropout=0.5,
                max_t=int(config['max_t']),
                max_s=int(config['max_s']),
                use_cls_token=bool(config['use_cls_token']),
                pool='cls' if bool(config['use_cls_token']) else 'max',
            ).to(config['device'])
        else:
            raise ValueError(f"Unsupported single-item feature shape: {tuple(x0.shape)}")

        weights = torch.tensor(config['class_weights'], dtype=torch.float32).to(config['device'])

        class FocalLoss(nn.Module):
            def __init__(self, weight=None, gamma: float = 2.0, label_smoothing: float = 0.0):
                super().__init__()
                self.register_buffer('weight', weight if weight is not None else None)
                self.gamma = float(gamma)
                self.label_smoothing = float(label_smoothing)

            def forward(self, logits, target):
                # logits: (B,K), target: (B,)
                ce = nn.functional.cross_entropy(
                    logits,
                    target,
                    weight=self.weight,
                    reduction='none',
                    label_smoothing=self.label_smoothing,
                )
                p = torch.softmax(logits, dim=1)
                pt = p.gather(1, target.view(-1, 1)).squeeze(1).clamp_min(1e-6)
                loss = ((1.0 - pt) ** self.gamma) * ce
                return loss.mean()

        if float(config.get('focal_gamma', 0.0)) and float(config['focal_gamma']) > 0:
            criterion = FocalLoss(weight=weights, gamma=float(config['focal_gamma']), label_smoothing=0.0)
        else:
            criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.02)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, threshold=1e-4
        )

        # in features mode we don't use AMP unless cuda
        scaler = torch.amp.GradScaler(enabled=config['amp'])

        best_acc = 0.0
        best_f1 = 0.0
        history = []
        patience = 10
        stale = 0

        def eval_loader(dloader):
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for seq, lbl in dloader:
                    seq = seq.to(config['device'])
                    lbl = torch.as_tensor(lbl, device=config['device'])
                    outputs = model(seq)
                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_labels.extend(lbl.cpu().numpy().tolist())
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            acc = 100.0 * float(np.mean(all_preds == all_labels)) if all_labels.size else 0.0
            report_dict = classification_report(all_labels, all_preds, labels=list(range(8)), output_dict=True, zero_division=0)
            macro_f1 = float(report_dict.get('macro avg', {}).get('f1-score', 0.0))
            return acc, macro_f1, all_labels, all_preds

        for epoch in range(1, config['epochs'] + 1):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for seq, lbl in train_loader:
                seq = seq.to(config['device'])
                lbl = torch.as_tensor(lbl, device=config['device'])
                optimizer.zero_grad(set_to_none=True)

                preds = model(seq)
                loss = criterion(preds, lbl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += float(loss.item())
                train_correct += int((preds.argmax(dim=1) == lbl).sum().item())
                train_total += int(lbl.numel())

            train_acc = 100.0 * train_correct / max(train_total, 1)
            avg_loss = train_loss / max(len(train_loader), 1)

            test_acc, macro_f1, all_labels, all_preds = eval_loader(test_loader)
            scheduler.step(macro_f1 if args.best_metric == 'macro_f1' else (test_acc / 100.0))

            print(f"Epoch {epoch:02d}/{config['epochs']} | loss={avg_loss:.4f} | train_acc={train_acc:.2f}% | test_acc={test_acc:.2f}% | macro_f1={macro_f1:.3f}")

            is_best = False
            if args.best_metric == 'macro_f1':
                if (macro_f1 > best_f1 + 1e-6) or (abs(macro_f1 - best_f1) < 1e-6 and test_acc > best_acc):
                    is_best = True
            else:
                if test_acc > best_acc + 1e-6:
                    is_best = True

            if is_best:
                best_acc = max(best_acc, test_acc)
                best_f1 = max(best_f1, macro_f1)
                torch.save(model.state_dict(), os.path.join(config['output_dir'], 'best_model.pth'))
                plot_confusion_matrix(all_labels, all_preds, [str(i) for i in range(8)], epoch, config['output_dir'])
                with open(os.path.join(config['output_dir'], 'report.txt'), 'w') as f:
                    f.write(classification_report(all_labels, all_preds, labels=list(range(8)), target_names=[str(i) for i in range(8)], zero_division=0))
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    print('[Early stopping] no improvement for', patience, 'epochs')
                    break

            history.append({'epoch': epoch, 'loss': avg_loss, 'train_acc': train_acc, 'test_acc': test_acc, 'macro_f1': macro_f1, 'best_test_acc': best_acc, 'best_macro_f1': best_f1})

        pd.DataFrame(history).to_csv(os.path.join(config['output_dir'], 'history.csv'), index=False)
        print(f"[Done] best acc {best_acc:.2f}% | best macro_f1 {best_f1:.3f}")
        return

    else:
        train_tf = transforms.Compose([
            transforms.Resize((config['img_size'], config['img_size'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.15, 0.15, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_tf = transforms.Compose([
            transforms.Resize((config['img_size'], config['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_set = UnsafeNetTemporalDataset('data', 'train', config['sequence_length'], train_tf)
        test_set = UnsafeNetTemporalDataset('data', 'test', config['sequence_length'], test_tf)

        # ----- Class weights (robust) -----
        labels = np.array([lbl for _, lbl in train_set.samples], dtype=int)
        num_classes = int(labels.max()) + 1 if labels.size else 8
        num_classes = max(num_classes, 8)
        class_counts = np.bincount(labels, minlength=num_classes)
        class_counts_safe = np.maximum(class_counts, 1)
        # inverse-frequency normalized to mean 1.0
        inv = labels.size / (num_classes * class_counts_safe)
        class_weights = (inv / inv.mean()).astype(np.float32)
        config['class_weights'] = class_weights.tolist()

        # Optional: weighted sampler to fight class imbalance
        sampler = None
        shuffle = True
        if config['use_sampler']:
            from torch.utils.data import WeightedRandomSampler
            sample_w = class_weights[labels]
            sampler = WeightedRandomSampler(weights=torch.as_tensor(sample_w, dtype=torch.double),
                                            num_samples=len(sample_w), replacement=True)
            shuffle = False

        train_loader = DataLoader(
            train_set,
            batch_size=config['batch_size'],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=(config['device'] == 'cuda'),
            persistent_workers=(args.num_workers > 0),
        )
        test_loader = DataLoader(
            test_set,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(config['device'] == 'cuda'),
            persistent_workers=(args.num_workers > 0),
        )

        model = YOLO_Temporal_Transformer(
            num_classes=8,
            nhead=8,
            num_layers=4,
            dropout=0.5,
            yolo_weights=config['yolo_weights'],
            feature_layer_idx=config['feature_layer_idx'],
        ).to(config['device'])

        weights = torch.tensor(config['class_weights'], dtype=torch.float32).to(config['device'])
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

        # ---- Optimizer: start with head only (backbone frozen) ----
        backbone_param_ids = {id(p) for p in model.get_backbone_parameters()}
        head_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]

        optimizer = optim.AdamW(head_params, lr=config['learning_rate'], weight_decay=0.02)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'],
            steps_per_epoch=len(train_loader),
            epochs=config['epochs']
        )

        for p in model.get_backbone_parameters():
            p.requires_grad = False

    scaler = torch.amp.GradScaler(enabled=config['amp'])

    best_acc = 0.0
    history = []
    patience = 8
    stale = 0

    def eval_loader(dloader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for seq, lbl in dloader:
                seq, lbl = seq.to(config['device']), lbl.to(config['device'])
                with torch.amp.autocast(device_type='cuda', enabled=config['amp']):
                    outputs = model(seq)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(lbl.cpu().numpy().tolist())
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = 100.0 * float(np.mean(all_preds == all_labels)) if all_labels.size else 0.0
        return acc, all_labels, all_preds

    for epoch in range(1, config['epochs'] + 1):
        if epoch == config['freeze_until_epoch'] + 1 and args.mode == 'video':
            for p in model.get_backbone_parameters():
                p.requires_grad = True
            # add backbone param group with smaller LR
            backbone_params = [p for p in model.get_backbone_parameters() if p.requires_grad]
            optimizer.add_param_group({'params': backbone_params, 'lr': config['learning_rate'] * config['backbone_lr_mult']})
            print("\n[!] UNFREEZING BACKBONE - entering full fine-tuning")

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for seq, lbl in train_loader:
            seq, lbl = seq.to(config['device']), lbl.to(config['device'])
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', enabled=config['amp']):
                preds = model(seq)
                loss = criterion(preds, lbl)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += float(loss.item())
            train_correct += int((preds.argmax(dim=1) == lbl).sum().item())
            train_total += int(lbl.numel())

        train_acc = 100.0 * train_correct / max(train_total, 1)
        avg_loss = train_loss / max(len(train_loader), 1)

        test_acc, all_labels, all_preds = eval_loader(test_loader)
        report_dict = classification_report(all_labels, all_preds, labels=list(range(8)), output_dict=True, zero_division=0)
        macro_f1 = float(report_dict.get('macro avg', {}).get('f1-score', 0.0))

        print(f"Epoch {epoch:02d}/{config['epochs']} | loss={avg_loss:.4f} | train_acc={train_acc:.2f}% | test_acc={test_acc:.2f}% | macro_f1={macro_f1:.3f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(config['output_dir'], 'best_model.pth'))
            plot_confusion_matrix(all_labels, all_preds, [str(i) for i in range(8)], epoch, config['output_dir'])
            with open(os.path.join(config['output_dir'], 'report.txt'), 'w') as f:
                f.write(classification_report(all_labels, all_preds, labels=list(range(8)), target_names=[str(i) for i in range(8)], zero_division=0))
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print('[Early stopping] no improvement for', patience, 'epochs')
                break

        history.append({'epoch': epoch, 'loss': avg_loss, 'train_acc': train_acc, 'test_acc': test_acc, 'macro_f1': macro_f1, 'best_test_acc': best_acc})

    pd.DataFrame(history).to_csv(os.path.join(config['output_dir'], 'history.csv'), index=False)
    print(f"[Done] best acc {best_acc:.2f}%")

if __name__ == '__main__':
    run_experiment()