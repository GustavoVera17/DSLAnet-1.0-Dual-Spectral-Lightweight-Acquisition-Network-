import torch
import torch.nn as nn
import torch.nn.functional as F

class CASSiPhysics(nn.Module):
    def __init__(self, mask_tensor, crf_np):
        super(CASSiPhysics, self).__init__()
        # Registramos la mÃ¡scara y la curva como 'buffers'. 
        # Esto asegura que vayan automÃ¡ticamente a la GPU sin ser entrenables.
        self.register_buffer('mask', mask_tensor) # [1, 1, H, W]
        self.register_buffer('crf', torch.from_numpy(crf_np)) # [3, 31]

    def shift_back(self, cassi_meas):
        """
        Paso 1: Convierte la imagen CASSI 2D [B, 1, H, W+30] 
        en el Cubo Ruidoso de 31 bandas [B, 31, H, W] para entrar a la red.
        """
        B, _, H, W_disp = cassi_meas.shape
        C = 31
        W = W_disp - C + 1
        
        cubo_inicial = torch.zeros((B, C, H, W), device=cassi_meas.device)
        for i in range(C):
            # Recortamos la ventana exacta para cada longitud de onda
            cubo_inicial[:, i, :, :] = cassi_meas[:, 0, :, i : i + W]
            
        return cubo_inicial

    def shift_forward(self, hsi_cube):
        """
        El Proceso FÃ­sico del Prisma: Dispersa el cubo 3D reconstruido [B, 31, H, W] 
        para crear una simulaciÃ³n CASSI 2D [B, 1, H, W+30].
        """
        B, C, H, W = hsi_cube.shape
        W_disp = W + C - 1
        
        # 1. Aplicar la mÃ¡scara del sistema (Coded Aperture)
        masked_cube = hsi_cube * self.mask
        
        # 2. DispersiÃ³n (Prisma)
        cassi_sim = torch.zeros((B, 1, H, W_disp), device=hsi_cube.device)
        for i in range(C):
            cassi_sim[:, 0, :, i : i + W] += masked_cube[:, i, :, :]
            
        return cassi_sim

    def project_rgb(self, hsi_cube):
        """
        El Proceso FÃ­sico de la CÃ¡mara de Color: Aplasta el cubo 3D [B, 31, H, W]
        a un RGB [B, 3, H, W] usando la Curva de Eficiencia CuÃ¡ntica (CRF).
        """
        B, C, H, W = hsi_cube.shape
        
        # Aplanamos el espacio para multiplicar pÃ­xel por pÃ­xel con la matriz de 3x31
        hsi_flat = hsi_cube.view(B, C, H * W) # [B, 31, H*W]
        
        # MultiplicaciÃ³n de matrices en PyTorch: (CRF) x (Cubo Aplanado)
        # crf.unsqueeze(0) permite que la misma curva se aplique a todo el batch
        rgb_sim_flat = torch.matmul(self.crf.unsqueeze(0), hsi_flat) # [B, 3, H*W]
        
        # Devolvemos la forma de imagen [B, 3, H, W]
        rgb_sim = rgb_sim_flat.view(B, 3, H, W)
        
        return rgb_sim

class SelfSupervisedLoss(nn.Module):
    def __init__(self, mask_tensor, crf_np):
        super(SelfSupervisedLoss, self).__init__()
        self.physics = CASSiPhysics(mask_tensor, crf_np)
        
    def forward(self, pred_cube, cassi_real, rgb_real, epoca_actual=1, max_epochs=1200):
        cassi_simulado = self.physics.shift_forward(pred_cube)
        loss_cassi = F.mse_loss(cassi_simulado, cassi_real)
        
        rgb_simulado = self.physics.project_rgb(pred_cube)
        loss_rgb = F.mse_loss(rgb_simulado, rgb_real)
        
        # Decaimiento de Lambda (Annealing): Empieza en 20.0 y baja hasta 1.0
        lambda_inicial = 20.0
        lambda_final = 1.0
        progreso = min(1.0, max(0.0, (epoca_actual - 1) / max(1, max_epochs - 1)))
        lambda_color = lambda_inicial - progreso * (lambda_inicial - lambda_final)
        
        # Spectral Smoothness Loss (Suavidad Espectral)
        loss_smoothness = torch.mean(torch.abs(pred_cube[:, 1:, :, :] - pred_cube[:, :-1, :, :]))
        lambda_smooth = 0.5
        
        loss_total = loss_cassi + (lambda_color * loss_rgb) + (lambda_smooth * loss_smoothness)
        
        return loss_total, loss_cassi, loss_rgb
