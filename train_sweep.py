import os
import argparse
import time
import datetime
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_metric

from utils.dataset_dual import CASSIDualDataset
from utils.metricas import calcular_psnr, calcular_sam
from utils.physics_loss import CASSiPhysics, SelfSupervisedLoss
from models.dslanet_universal import DSLAnet_Universal

def get_args():
    parser = argparse.ArgumentParser(description="Sweep Trainer for DSLAnet Ablation")
    parser.add_argument('--variant', type=str, required=True, choices=['baseline', 'spatcab', 'speccab', 'fullcab'])
    parser.add_argument('--blocks', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=16)
    return parser.parse_args()

def main():
    args = get_args()
    
    use_spatcab = args.variant in ['spatcab', 'fullcab']
    use_speccab = args.variant in ['speccab', 'fullcab']
    
    experiment_name = f"{args.variant}_{args.blocks}blocks"
    
    DIR_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", experiment_name)
    DIR_DASHBOARDS = os.path.join(DIR_BASE, "dashboards")
    DIR_PESOS = os.path.join(DIR_BASE, "pesos")
    os.makedirs(DIR_DASHBOARDS, exist_ok=True)
    os.makedirs(DIR_PESOS, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    RUTA_TRAIN = r"C:\CODE2026\CASSIproy2026\CASSIproy2026\CAVE_Raw\fortrain"
    RUTA_TEST  = r"C:\CODE2026\CASSIproy2026\CASSIproy2026\CAVE_Raw\forval"
    
    dataset_train = CASSIDualDataset(RUTA_TRAIN, patch_size=256, num_patches_per_img=10, is_train=True)
    dataset_val   = CASSIDualDataset(RUTA_TEST, patch_size=256, num_patches_per_img=1, is_train=False)

    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    loader_val   = DataLoader(dataset_val, batch_size=1, shuffle=False)

    modelo = DSLAnet_Universal(num_blocks=args.blocks, use_spatcab=use_spatcab, use_speccab=use_speccab).to(device)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs*0.3), gamma=0.2)

    mascara_sistema = dataset_train.get_mask().to(device)
    crf_np = dataset_train.crf_np
    fisica = CASSiPhysics(mascara_sistema, crf_np).to(device)
    
    mascara_val = dataset_val.get_mask().to(device)
    fisica_val = CASSiPhysics(mascara_val, crf_np).to(device)
    
    juez_supremo = SelfSupervisedLoss(mascara_sistema, crf_np).to(device)
    scaler = torch.amp.GradScaler('cuda')

    historial = {'loss_t': [], 'loss_c': [], 'loss_g': [], 'psnr': [], 'sam': [], 'ssim': []}
    mejor_psnr = 0.0
    
    master_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "master_ablation_log.csv")
    if not os.path.exists(master_csv):
        with open(master_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Variant', 'Blocks', 'Epoch', 'Loss', 'PSNR', 'SSIM', 'SAM'])
            
    for epoca in range(1, args.epochs + 1):
        modelo.train()
        loss_epoch = 0.0; loss_gb_ep = 0.0; loss_cb_ep = 0.0
        loop = tqdm(loader_train, desc=f"{experiment_name} [{epoca}/{args.epochs}]", leave=False)
        optimizer.zero_grad() 
        
        for i, (cassi_real, rgb_real, _) in enumerate(loop):
            cassi_real = cassi_real.to(device)
            rgb_real = rgb_real.to(device)
            with torch.amp.autocast('cuda'):
                cassi_shiftback = fisica.shift_back(cassi_real)
                pred_3d = modelo(cassi_shiftback, rgb_real)
                loss_t, loss_g, loss_c = juez_supremo(pred_3d, cassi_real, rgb_real, epoca, args.epochs)
            
            scaler.scale(loss_t).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() 
            
            loss_epoch += loss_t.item(); loss_gb_ep += loss_g.item(); loss_cb_ep += loss_c.item()
            loop.set_postfix(LossT=f"{(loss_epoch/(i+1)):.4f}")
            
        scheduler.step()
        historial['loss_t'].append(loss_epoch/len(loader_train))
        historial['loss_c'].append(loss_cb_ep/len(loader_train))
        historial['loss_g'].append(loss_gb_ep/len(loader_train))
        
        modelo.eval()
        p_tot = 0.0; sm_tot = 0.0; ss_tot = 0.0
        vis_cassi = None; vis_rgb = None; vis_gt = None; vis_pred = None
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for cassi_v, rgb_v, gt_v in loader_val:
                    cassi_v = cassi_v.to(device)
                    rgb_v = rgb_v.to(device)
                    gt_v = gt_v.to(device)
                    pred_v = modelo(fisica_val.shift_back(cassi_v), rgb_v)
                    
                    p_tot += calcular_psnr(pred_v, gt_v)
                    sm_tot += calcular_sam(pred_v.contiguous(), gt_v.contiguous())
                    p_np = torch.clamp(pred_v[0],0,1).cpu().numpy().transpose(1,2,0)
                    g_np = torch.clamp(gt_v[0],0,1).cpu().numpy().transpose(1,2,0)
                    ss_tot += ssim_metric(g_np, p_np, data_range=1.0, channel_axis=-1)
                    
                    vis_cassi = cassi_v; vis_rgb = rgb_v; vis_gt = gt_v; vis_pred = pred_v
                    
        p_med = p_tot/len(loader_val)
        sm_med = sm_tot/len(loader_val)
        ss_med = ss_tot/len(loader_val)
        historial['psnr'].append(p_med); historial['sam'].append(sm_med); historial['ssim'].append(ss_med)
        
        if p_med > mejor_psnr:
            mejor_psnr = p_med
            torch.save(modelo.state_dict(), os.path.join(DIR_PESOS, "mejor_modelo.pth"))
            
        # Logging to master CSV every 10 epochs
        if epoca % 10 == 0 or epoca == args.epochs:
            with open(master_csv, 'a', newline='') as f:
                csv.writer(f).writerow([args.variant, args.blocks, epoca, f"{historial['loss_t'][-1]:.4f}", f"{p_med:.2f}", f"{ss_med:.4f}", f"{sm_med:.2f}"])
        
        # Dashboard generation
        if epoca % 50 == 0 or epoca == args.epochs:
            fig, axs = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(f"{experiment_name.upper()} | Epoch: {epoca} | PSNR: {p_med:.2f}", fontsize=18, fontweight='bold')
            axs[0, 0].imshow(vis_cassi[0, 0].cpu().numpy(), cmap='gray'); axs[0, 0].set_title("CASSI PAN"); axs[0, 0].axis('off')
            rgb_plot = np.clip(vis_rgb[0].cpu().numpy().transpose(1,2,0), 0, 1)
            axs[0, 1].imshow(rgb_plot); axs[0, 1].set_title("RGB (Ground Truth)"); axs[0, 1].axis('off')
            axs[0, 2].imshow(vis_gt[0, 15].cpu().numpy(), cmap='gray', vmin=0, vmax=1); axs[0, 2].set_title("Band 15 (GT)"); axs[0, 2].axis('off')
            axs[0, 3].imshow(vis_pred[0, 15].cpu().numpy(), cmap='gray', vmin=0, vmax=1); axs[0, 3].set_title("Band 15 (Pred)"); axs[0, 3].axis('off')
            
            axs[1, 0].plot(historial['loss_t'], 'k-', label='Total'); axs[1, 0].set_title("Loss")
            axs[1, 1].plot(historial['psnr'], 'b-'); axs[1, 1].set_title("PSNR (dB)")
            axs[1, 2].plot(historial['ssim'], 'g-'); axs[1, 2].set_title("SSIM")
            axs[1, 3].plot(historial['sam'], 'purple'); axs[1, 3].set_title("SAM (Â°)")
            
            plt.tight_layout()
            dash_name = f"dashboard_{experiment_name}_ep{epoca}.png"
            plt.savefig(os.path.join(DIR_DASHBOARDS, dash_name), bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    main()
