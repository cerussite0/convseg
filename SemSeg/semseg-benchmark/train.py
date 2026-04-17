import argparse
import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from convseg_net.model import ConvSegNet
from training.losses import CrossEntropyDiceLoss
from evaluation.iou import compute_iou
from evaluation.dice import compute_dice
from data.loaders.ade20k import ADE20KDataset
from data.loaders.cityscapes import CityscapesDataset
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler()])

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate and Train deep segmentation algorithms.')
    parser.add_argument('--model', type=str, default='convseg_net', choices=['convseg_net'], help='Select model to train.')
    parser.add_argument('--encoder', type=str, default='resnet34', help='Encoder backbone for the model (e.g. resnet34, resnet50, efficientnet-b4). Pretrained ImageNet weights are downloaded automatically.')
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['ade20k', 'cityscapes'], help='Select Dataset wrapper.')
    parser.add_argument('--data-root', type=str, default='./data', help='Path to dataset root directory (Default: ./data).')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--weight-decay', type=float, default=1e-05, help='L2 regularization (weight decay).')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on.')
    return parser.parse_args()

def train(model, train_loader, val_loader, criterion, optimizer, args, num_classes):
    device = torch.device(args.device)
    model = model.to(device)
    out_dir = f'runs/{args.model}_{args.dataset}'
    weights_dir = os.path.join(out_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    history = {'train_loss': [], 'val_loss': [], 'val_miou': [], 'val_dice': []}
    best_miou = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_total = 0
        logging.info(f'Epoch {epoch + 1}/{args.epochs}')
        for batch in tqdm(train_loader, desc='Training'):
            inputs, target = (batch['img'].to(device), batch['mask'].to(device))
            predict = model(inputs)
            optimizer.zero_grad()
            loss = criterion(predict, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            n_total += inputs.size(0)
        train_loss /= n_total
        model.eval()
        val_loss, total_iou, total_dice, num_samples = (0.0, 0.0, 0.0, 0)
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                inputs, target = (batch['img'].to(device), batch['mask'].to(device))
                predict = model(inputs)
                loss = criterion(predict, target)
                pred_class = predict.argmax(dim=1)
                _, miou = compute_iou(pred_class, target, num_classes=num_classes, ignore_index=255)
                _, mdice = compute_dice(pred_class, target, num_classes=num_classes, ignore_index=255)
                val_loss += loss.item() * inputs.size(0)
                if not torch.isnan(miou) and (not torch.isnan(mdice)):
                    total_iou += miou.item() * inputs.size(0)
                    total_dice += mdice.item() * inputs.size(0)
                    num_samples += inputs.size(0)
        val_loss /= len(val_loader.dataset)
        val_miou = total_iou / num_samples if num_samples > 0 else 0.0
        val_dice = total_dice / num_samples if num_samples > 0 else 0.0
        logging.info(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} | Val mDice: {val_dice:.4f}')
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)
        history['val_dice'].append(val_dice)
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), os.path.join(weights_dir, 'best.pt'))
            logging.info(f'Saved best model with mIoU: {best_miou:.4f}')
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_miou': best_miou}, os.path.join(weights_dir, 'last.pt'))
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(weights_dir, f'epoch_{epoch + 1}.pt')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_miou': best_miou}, ckpt_path)
            logging.info(f'Saved periodic checkpoint: {ckpt_path}')
        with open(os.path.join(out_dir, 'training_log.json'), 'w') as f:
            json.dump(history, f, indent=2)
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss')
    ax1.plot(epochs_range, history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{args.model.upper()} — Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs_range, history['val_miou'], label='Val mIoU', color='green')
    ax2.plot(epochs_range, history['val_dice'], label='Val Dice', color='purple')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title(f'{args.model.upper()} — Validation mIoU/Dice')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=150)
    plt.close(fig)
    logging.info(f'Training curves saved to {out_dir}/training_curves.png')
    logging.info(f'Training log saved to {out_dir}/training_log.json')
    logging.info(f'Best mIoU: {best_miou:.4f}')

def main():
    args = get_args()
    logging.info(f'Loading {args.dataset} dataset from {args.data_root}...')
    if args.dataset == 'ade20k':
        trainset = ADE20KDataset(root=args.data_root, split='train')
        valset = ADE20KDataset(root=args.data_root, split='val')
        num_classes = 150
    elif args.dataset == 'cityscapes':
        trainset = CityscapesDataset(root=args.data_root, split='train')
        valset = CityscapesDataset(root=args.data_root, split='val')
        num_classes = 19
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=valset, batch_size=args.batch_size, shuffle=False)
    logging.info(f'Initializing {args.model} for {num_classes} classes...')
    if args.model == 'convseg_net':
        model = ConvSegNet(num_classes=num_classes)
    else:
        raise ValueError(f'Unknown model: {args.model}')
    criterion = CrossEntropyDiceLoss(ignore_index=255)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    logging.info(f'Starting training on {args.device} for {args.epochs} epochs...')
    train(model, train_loader, val_loader, criterion, optimizer, args, num_classes)
if __name__ == '__main__':
    main()