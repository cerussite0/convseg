import argparse
import os
import json
import torch
import numpy as np
import logging
from tqdm import tqdm
import random
from torch.utils.data import DataLoader, Subset
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler('eval_reports.log', mode='a'), logging.StreamHandler()])
from data.loaders.cityscapes import CityscapesDataset
from data.loaders.ade20k import ADE20KDataset
from methods.classical.threshold import ThresholdSegmentation
from methods.classical.graph_cut import GraphCutSegmentation
from methods.classical.region import RegionSegmentation
from methods.classical.edge import EdgeSegmentation
from methods.ml.kmeans import KMeansSegmentation
from methods.ml.gmm import GMMSegmentation
from methods.ml.svm import SVMSegmentation
from utils.visualize import save_segmentation_maps
from evaluation.iou import compute_iou
from evaluation.dice import compute_dice
from evaluation.pixel_acc import compute_pixel_accuracy
from evaluation.mappings import map_clusters_to_classes

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate threshold segmentation algorithms.')
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['cityscapes', 'ade20k'], help='Select Dataset wrapper.')
    parser.add_argument('--data-root', type=str, default='./data', help='Path to dataset root directory (Default: ./data).')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='Dataset split to evaluate on.')
    parser.add_argument('--num-samples', type=int, default=None, help='Maximum number of samples to evaluate (default: all).')
    parser.add_argument('--batch-size', type=int, default=1, help='Evaluation batch size.')
    parser.add_argument('--method', type=str, default='otsu', choices=['otsu', 'global', 'graph_cut', 'region', 'kmeans', 'edge', 'gmm', 'svm', 'convseg_net', 'fcn', 'segformer', 'deeplabv3'], help='Segmentation method.')
    parser.add_argument('--global-thresh', type=int, default=127, help='Global threshold value (used iff method=global).')
    parser.add_argument('--visualize', action='store_true', help='Save visualization maps.')
    parser.add_argument('--visualize-only', action='store_true', help='Only generate visualization maps for selected indices; skip metric evaluation.')
    parser.add_argument('--vis-count', type=int, default=10, help='Number of random samples to visualize.')
    parser.add_argument('--vis-seed', type=int, default=42, help='Seed for bounding deterministic visualizations randomly scattering over dataset.')
    parser.add_argument('--vis-complex', action='store_true', help='Forces the visualizer array to specifically isolate images with >= 4 unique semantic classes!')
    parser.add_argument('--vis-indices-file', type=str, default=None, help='Optional path to save/load fixed visualization indices for cross-method comparison.')
    return parser.parse_args()

def main():
    args = get_args()
    if args.visualize_only and (not args.visualize):
        logging.info('--visualize-only set without --visualize; enabling --visualize automatically.')
        args.visualize = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Executing Evaluation ({args.method}) tracking on {device}...')
    if args.dataset == 'cityscapes':
        logging.info(f'Initializing Cityscapes loader inside {args.data_root} ...')
        dataset = CityscapesDataset(root=args.data_root, split=args.split)
    elif args.dataset == 'ade20k':
        logging.info(f'Initializing ADE20K loader inside {args.data_root} ...')
        dataset = ADE20KDataset(root=args.data_root, split=args.split)
    dataset_size = len(dataset)
    if dataset_size == 0:
        logging.error('Dataset empty. Check your data root path.')
        return
    if args.num_samples is not None:
        if args.num_samples <= 0:
            logging.error('--num-samples must be a positive integer.')
            return
        effective_num_samples = min(args.num_samples, dataset_size)
        if effective_num_samples < dataset_size:
            dataset = Subset(dataset, range(effective_num_samples))
            logging.info(f'Limiting evaluation to first {effective_num_samples} samples out of {dataset_size} due to --num-samples.')
        else:
            logging.info(f'--num-samples ({args.num_samples}) exceeds dataset size ({dataset_size}); evaluating all {dataset_size} samples.')
    else:
        effective_num_samples = dataset_size
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    if args.method in ['fcn', 'segformer', 'deeplabv3']:
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from model_utils import load_model
        from mmseg_inference import MODELS
        chkpt = MODELS.get(args.dataset, {}).get(args.method, {}).get('checkpoint', None)
        model = load_model(args.method, args.dataset, checkpoint=chkpt, device=device)
        model.eval()
    elif args.method == 'convseg_net':
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from convseg_net.model import ConvSegNet
        from mmseg_inference import load_convseg_checkpoint, MODELS
        chkpt = MODELS.get(args.dataset, {}).get('convseg', {}).get('checkpoint', None)
        model = ConvSegNet(num_classes=150 if args.dataset == 'ade20k' else 19)
        model = load_convseg_checkpoint(model, chkpt, device)
        model = model.to(device)
        model.eval()
    elif args.method in ['otsu', 'global']:
        model = ThresholdSegmentation(method=args.method, global_thresh=args.global_thresh).to(device)
    elif args.method == 'graph_cut':
        model = GraphCutSegmentation().to(device)
    elif args.method == 'region':
        model = RegionSegmentation().to(device)
    elif args.method == 'kmeans':
        model = KMeansSegmentation().to(device)
    elif args.method == 'gmm':
        model = GMMSegmentation().to(device)
    elif args.method == 'svm':
        model = SVMSegmentation().to(device)
    elif args.method == 'edge':
        model = EdgeSegmentation().to(device)
    vis_target_indices = set()
    if args.visualize:
        random.seed(args.vis_seed)
        sample_scope = 'full' if args.num_samples is None else f'n{effective_num_samples}'
        loaded_fixed_indices = None
        if args.vis_indices_file and os.path.exists(args.vis_indices_file):
            try:
                if args.vis_indices_file.endswith('.npy'):
                    loaded = np.load(args.vis_indices_file).tolist()
                else:
                    with open(args.vis_indices_file, 'r') as f:
                        payload = json.load(f)
                    loaded = payload.get('indices', payload)
                loaded_fixed_indices = [int(i) for i in loaded if isinstance(i, (int, float)) and 0 <= int(i) < len(dataset)]
                if args.vis_count > 0:
                    loaded_fixed_indices = loaded_fixed_indices[:args.vis_count]
                if len(loaded_fixed_indices) > 0:
                    vis_target_indices = set(loaded_fixed_indices)
                    logging.info(f'Loaded {len(vis_target_indices)} fixed visualization indices from {args.vis_indices_file}: {sorted(vis_target_indices)}')
                else:
                    logging.warning(f'No valid visualization indices found in {args.vis_indices_file}; falling back to dynamic selection.')
            except Exception as e:
                logging.warning(f'Failed loading --vis-indices-file ({args.vis_indices_file}): {e}. Falling back to dynamic selection.')
        if len(vis_target_indices) == 0 and args.vis_complex:
            complex_cache_path = os.path.join(args.data_root, f'{args.dataset}_{args.split}_{sample_scope}_complex_indices.npy')
            if os.path.exists(complex_cache_path):
                logging.info(f'Loading cached complex indices from {complex_cache_path}...')
                complex_idx = np.load(complex_cache_path).tolist()
            else:
                logging.info('Scanning dataset purely isolating highly complex semantic scenes (>= 4 IDs)...')
                complex_idx = []
                for i in tqdm(range(len(dataset)), desc='Complexity Scan'):
                    target_mask = dataset[i]['mask']
                    valids = [c.item() for c in torch.unique(target_mask) if c.item() != 255]
                    if len(valids) >= 4:
                        complex_idx.append(i)
                np.save(complex_cache_path, complex_idx)
                logging.info(f'Saved complex indices cache to {complex_cache_path}!')
            vis_target_indices = set(random.sample(complex_idx, min(args.vis_count, len(complex_idx))))
            logging.info(f'Locked {len(vis_target_indices)} deterministic complex samples dynamically mapped!')
        elif len(vis_target_indices) == 0:
            vis_target_indices = set(random.sample(range(len(dataset)), min(args.vis_count, len(dataset))))
        if args.vis_indices_file and (not os.path.exists(args.vis_indices_file)):
            try:
                os.makedirs(os.path.dirname(os.path.abspath(args.vis_indices_file)), exist_ok=True)
                sorted_indices = sorted(vis_target_indices)
                if args.vis_indices_file.endswith('.npy'):
                    np.save(args.vis_indices_file, np.array(sorted_indices, dtype=np.int64))
                else:
                    payload = {'dataset': args.dataset, 'split': args.split, 'num_samples': effective_num_samples, 'vis_complex': args.vis_complex, 'vis_seed': args.vis_seed, 'vis_count': args.vis_count, 'indices': sorted_indices}
                    with open(args.vis_indices_file, 'w') as f:
                        json.dump(payload, f, indent=2)
                logging.info(f'Saved {len(vis_target_indices)} visualization indices to {args.vis_indices_file}: {sorted_indices}')
            except Exception as e:
                logging.warning(f'Failed saving visualization indices to {args.vis_indices_file}: {e}')
    if args.visualize_only:
        if len(vis_target_indices) == 0:
            logging.warning('No visualization indices selected; nothing to render.')
            return
        logging.info(f'Running visualization-only mode for {len(vis_target_indices)} selected samples...')
        selected_indices = sorted(vis_target_indices)
        vis_dataset = Subset(dataset, selected_indices)
        vis_loader = DataLoader(vis_dataset, batch_size=args.batch_size, shuffle=False)
        vis_images = []
        vis_targets = []
        vis_preds = []
        with torch.no_grad():
            for batch in tqdm(vis_loader, total=len(vis_loader), desc='Visualize Only'):
                img = batch['img'].to(device)
                target = batch['mask'].to(device)
                pred = model(img)
                if args.method in ['otsu', 'global', 'edge']:
                    target_eval = ((target > 0) & (target != 255)).long()
                    target_eval[target == 255] = 255
                    pred_eval = pred
                elif args.method in ['graph_cut', 'region', 'kmeans', 'gmm', 'svm']:
                    target_eval = target
                    pred_eval = map_clusters_to_classes(pred, target, ignore_index=255)
                elif args.method in ['fcn', 'segformer', 'deeplabv3', 'convseg_net']:
                    target_eval = target
                    if pred.shape[2:] != target.shape[-2:]:
                        import torch.nn.functional as F
                        pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
                    pred_eval = pred.argmax(dim=1)
                for b in range(pred_eval.shape[0]):
                    vis_images.append(img[b].cpu())
                    vis_targets.append(target_eval[b].cpu())
                    vis_preds.append(pred_eval[b].cpu())
        if len(vis_images) > 0:
            save_dir = os.path.join('.', 'results', f'{args.dataset}_{args.method}')
            save_segmentation_maps(vis_images, vis_targets, vis_preds, save_dir, prefix='vis', max_samples=len(vis_images), dataset_name=args.dataset)
            logging.info(f'Visualization-only outputs saved to {save_dir}')
        return
    total_iou = 0.0
    total_dice = 0.0
    total_acc = 0.0
    num_samples = 0
    vis_images = []
    vis_targets = []
    vis_preds = []
    global_idx = 0
    logging.info(f'Starting Evaluation Iteration for {args.dataset}...')
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            img = batch['img'].to(device)
            target = batch['mask'].to(device)
            pred = model(img)
            if args.method in ['otsu', 'global', 'edge']:
                target_eval = ((target > 0) & (target != 255)).long()
                target_eval[target == 255] = 255
                pred_eval = pred
                eval_classes = 2
            elif args.method in ['graph_cut', 'region', 'kmeans', 'gmm', 'svm']:
                target_eval = target
                pred_eval = map_clusters_to_classes(pred, target, ignore_index=255)
                eval_classes = 256
            elif args.method in ['fcn', 'segformer', 'deeplabv3', 'convseg_net']:
                target_eval = target
                if pred.shape[2:] != target.shape[-2:]:
                    import torch.nn.functional as F
                    pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
                pred_eval = pred.argmax(dim=1)
                eval_classes = 150 if args.dataset == 'ade20k' else 19
            if args.visualize:
                for b in range(pred_eval.shape[0]):
                    if global_idx + b in vis_target_indices:
                        vis_images.append(img[b].cpu())
                        vis_targets.append(target_eval[b].cpu())
                        vis_preds.append(pred_eval[b].cpu())
            global_idx += img.shape[0]
            _, miou = compute_iou(pred_eval, target_eval, num_classes=eval_classes, ignore_index=255)
            _, mdice = compute_dice(pred_eval, target_eval, num_classes=eval_classes, ignore_index=255)
            acc = compute_pixel_accuracy(pred_eval, target_eval, ignore_index=255)
            if not torch.isnan(miou) and (not torch.isnan(mdice)):
                total_iou += miou.item()
                total_dice += mdice.item()
                total_acc += acc.item()
                num_samples += 1
    if num_samples > 0:
        final_miou = total_iou / num_samples
        final_mdice = total_dice / num_samples
        final_acc = total_acc / num_samples
        logging.info('\n=== Evaluation Results ===')
        logging.info(f'Algorithm:       {args.method.upper()}')
        logging.info(f'Dataset:         {args.dataset.upper()} ({args.split})')
        logging.info(f'mIoU:            {final_miou:.4f}')
        logging.info(f'mDice:           {final_mdice:.4f}')
        logging.info(f'Pixel Accuracy:  {final_acc:.4f}')
    else:
        logging.warning('Completed loop, but no valid samples were evaluated. Please check dataset masks.')
    if args.visualize and len(vis_images) > 0:
        logging.info('Saving visualization maps...')
        save_dir = os.path.join('.', 'results', f'{args.dataset}_{args.method}')
        save_segmentation_maps(vis_images, vis_targets, vis_preds, save_dir, prefix='vis', max_samples=len(vis_images), dataset_name=args.dataset)
        logging.info(f'Visualizations saved to {save_dir}')
if __name__ == '__main__':
    main()