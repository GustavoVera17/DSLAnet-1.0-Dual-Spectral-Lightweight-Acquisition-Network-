import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

class CASSIDualDataset(Dataset):
    def __init__(self, root_dir, patch_size=256, num_patches_per_img=10, is_train=True):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.num_patches = num_patches_per_img
        self.is_train = is_train
        
        self.image_folders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
        
        # 1. LA MÃSCARA DEL SISTEMA (Fija para la resoluciÃ³n del parche)
        # Se guarda como tensor para que coincida con el entorno de PyTorch
        np.random.seed(42) # Semilla fija para que la mÃ¡scara no cambie entre ejecuciones
        self.mask_np = np.random.binomial(1, 0.5, (patch_size, patch_size)).astype(np.float32)
        self.mask_tensor = torch.from_numpy(self.mask_np).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

        # 2. LA MATRIZ DE LA CÃMARA RGB (CAVE CRF)
        self.crf_np = np.array([
            # Canal ROJO
            [0.0073, 0.0326, 0.1146, 0.2238, 0.2319, 0.1408, 0.0545, 0.0063, 0.0016, 0.0016,
             0.0020, 0.0049, 0.0163, 0.0458, 0.1065, 0.2088, 0.3516, 0.5284, 0.7042, 0.8359,
             0.8876, 0.8407, 0.7186, 0.5513, 0.3840, 0.2458, 0.1472, 0.0844, 0.0470, 0.0253, 0.0135],
            # Canal VERDE
            [0.0001, 0.0004, 0.0016, 0.0039, 0.0069, 0.0125, 0.0232, 0.0438, 0.0841, 0.1417,
             0.2114, 0.3160, 0.4578, 0.6127, 0.7303, 0.7711, 0.7259, 0.6094, 0.4566, 0.2974,
             0.1691, 0.0863, 0.0410, 0.0188, 0.0084, 0.0037, 0.0016, 0.0007, 0.0003, 0.0001, 0.0000],
            # Canal AZUL
            [0.0336, 0.1558, 0.5847, 1.2588, 1.4883, 1.1396, 0.6908, 0.3546, 0.1584, 0.0658,
             0.0267, 0.0099, 0.0038, 0.0015, 0.0005, 0.0002, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        ], dtype=np.float32)
        # Normalizamos la curva CRF para que la suma mÃ¡xima no queme la imagen
        self.crf_np = self.crf_np / np.max(np.sum(self.crf_np, axis=1))

        # 3. PRE-CARGA DE CUBOS A LA RAM
        self.cubes_cache = []
        print(f"Pre-cargando {len(self.image_folders)} imÃ¡genes CAVE. Esto puede tardar...")
        # Â¡AÃ±adimos tqdm aquÃ­ para ver la barra de progreso!
        for folder in tqdm(self.image_folders, desc="Subiendo a RAM"):
            cube = self._load_cave_cube(folder)
            self.cubes_cache.append(cube)
        print("Â¡Precarga completada!")

    def _load_cave_cube(self, folder_path):
        search_pattern = os.path.join(folder_path, '**', '*.png')
        all_pngs = sorted(glob.glob(search_pattern, recursive=True))
        
        band_files = []
        for f in all_pngs:
            filename = os.path.basename(f)
            name_no_ext = os.path.splitext(filename)[0]
            if name_no_ext[-2:].isdigit():
                band_files.append(f)
                
        band_files = sorted(band_files)[:31]
        
        if len(band_files) != 31:
            raise ValueError(f"Error en {folder_path}: Se encontraron {len(band_files)} bandas.")
        
        cube = []
        for file in band_files:
            img = Image.open(file)
            img_array = np.array(img, dtype=np.float32)
            
            if img_array.ndim == 3:
                img_array = img_array[:, :, 0]
                
            max_val = 65535.0 if np.max(img_array) > 255.0 else 255.0
            img_array = img_array / max_val
            cube.append(img_array)
            
        return np.stack(cube, axis=2) # [H, W, 31]
    
    def _simulate_cassi(self, hsi_patch):
        """Simula la dispersiÃ³n sobre el parche especÃ­fico"""
        H, W, C = hsi_patch.shape
        masked_patch = hsi_patch * self.mask_np[:, :, np.newaxis]
        
        cassi_meas = np.zeros((H, W + C - 1), dtype=np.float32)
        for i in range(C):
            cassi_meas[:, i:(i + W)] += masked_patch[:, :, i]
            
        return cassi_meas

    def _simulate_rgb(self, hsi_patch):
        """Simula la captura de la cÃ¡mara RGB usando la CRF de CAVE"""
        H, W, C = hsi_patch.shape
        hsi_flat = hsi_patch.reshape(-1, C).T # [31, H*W]
        rgb_flat = np.dot(self.crf_np, hsi_flat) # [3, 31] x [31, H*W] = [3, H*W]
        rgb_patch = rgb_flat.T.reshape(H, W, 3) # [H, W, 3]
        return rgb_patch

    def __len__(self):
        return len(self.image_folders) * self.num_patches

    def __getitem__(self, idx):
        img_idx = idx // self.num_patches
        full_cube = self.cubes_cache[img_idx]
        
        max_h = full_cube.shape[0] - self.patch_size
        max_w = full_cube.shape[1] - self.patch_size
        
        # 1. ExtracciÃ³n del parche limpio (Ground Truth real)
        rng = np.random.RandomState(idx) 
        start_h = rng.randint(0, max_h)
        start_w = rng.randint(0, max_w)
        patch_cube = full_cube[start_h : start_h + self.patch_size, start_w : start_w + self.patch_size, :]
        
        # 2. SimulaciÃ³n de los dos flujos de hardware (Xie et al.)
        patch_rgb = self._simulate_rgb(patch_cube)
        patch_cassi = self._simulate_cassi(patch_cube)
        
        # 3. ConversiÃ³n a Tensores de PyTorch
        tensor_cube = torch.from_numpy(patch_cube).permute(2, 0, 1)   # [31, 256, 256]
        tensor_rgb = torch.from_numpy(patch_rgb).permute(2, 0, 1)     # [3, 256, 256]
        tensor_cassi = torch.from_numpy(patch_cassi).unsqueeze(0)     # [1, 256, 286]
        
        return tensor_cassi, tensor_rgb, tensor_cube

    def get_mask(self):
        """Devuelve la mÃ¡scara para inicializar la funciÃ³n de pÃ©rdida"""
        return self.mask_tensor

# ==========================================
# ZONA DE PRUEBA (EJECUTABLE)
# ==========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # AsegÃºrate de cambiar esto a tu ruta real de "fortest" o "fortrain"
    RUTA_PRUEBA = r"C:\CODE2026\CASSIproy2026\CASSIproy2026\Yamawaki_Rep\dataset\fortest"
    
    try:
        print("Iniciando prueba del Dataloader Dual...")
        dataset = CASSIDualDataset(root_dir=RUTA_PRUEBA, patch_size=256, num_patches_per_img=1)
        
        # Extraemos un paquete de datos
        cassi, rgb, cube = dataset[0]
        
        print("\n--- DIMENSIONES GENERADAS ---")
        print(f"Cubo HSI (Ground Truth) : {cube.shape} -> DeberÃ­a ser [31, 256, 256]")
        print(f"Imagen RGB Simulada     : {rgb.shape}  -> DeberÃ­a ser [3, 256, 256]")
        print(f"Imagen CASSI Compresiva : {cassi.shape}  -> DeberÃ­a ser [1, 256, 286]")
        
        # Extraemos la mÃ¡scara del sistema para dÃ¡rsela a los jueces mÃ¡s adelante
        mascara_sistema = dataset.get_mask()
        print(f"MÃ¡scara del Sistema     : {mascara_sistema.shape} -> DeberÃ­a ser [1, 1, 256, 256]")
        
        # Opcional: Graficamos para verificar visualmente que no hay errores de Ã­ndices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.set_title("SimulaciÃ³n CÃ¡mara RGB")
        # Normalizamos solo para visualizar mejor (el RGB puede superar 1.0 por la matriz)
        rgb_plot = rgb.permute(1, 2, 0).numpy()
        rgb_plot = np.clip(rgb_plot / rgb_plot.max(), 0, 1)
        ax1.imshow(rgb_plot)
        ax1.axis('off')
        
        ax2.set_title("SimulaciÃ³n CÃ¡mara CASSI")
        ax2.imshow(cassi[0].numpy(), cmap='gray')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        print("\nÂ¡Prueba finalizada exitosamente!")
        
    except Exception as e:
        print(f"\n[!] Error en la prueba: {e}")
        print("Verifica que la ruta apunte a una carpeta con imágenes CAVE válidas.")


# ==========================================
# HARVARD DATASET LOADER
# ==========================================
class HarvardDualDataset(Dataset):
    def __init__(self, root_dir, patch_size=256, num_patches_per_img=10, is_train=True):
        import scipy.io
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.num_patches = num_patches_per_img
        self.is_train = is_train
        
        self.mat_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.mat')]
        
        np.random.seed(42)
        if self.is_train:
            self.mask_np = np.random.binomial(1, 0.5, (patch_size, patch_size)).astype(np.float32)
            self.mask_tensor = torch.from_numpy(self.mask_np).unsqueeze(0).unsqueeze(0)
        else:
            self.mask_np = None
            self.mask_tensor = None
            
        self.crf_np = np.array([
            [0.0073, 0.0326, 0.1146, 0.2238, 0.2319, 0.1408, 0.0545, 0.0063, 0.0016, 0.0016,
             0.0020, 0.0049, 0.0163, 0.0458, 0.1065, 0.2088, 0.3516, 0.5284, 0.7042, 0.8359,
             0.8876, 0.8407, 0.7186, 0.5513, 0.3840, 0.2458, 0.1472, 0.0844, 0.0470, 0.0253, 0.0135],
            [0.0001, 0.0004, 0.0016, 0.0039, 0.0069, 0.0125, 0.0232, 0.0438, 0.0841, 0.1417,
             0.2114, 0.3160, 0.4578, 0.6127, 0.7303, 0.7711, 0.7259, 0.6094, 0.4566, 0.2974,
             0.1691, 0.0863, 0.0410, 0.0188, 0.0084, 0.0037, 0.0016, 0.0007, 0.0003, 0.0001, 0.0000],
            [0.0336, 0.1558, 0.5847, 1.2588, 1.4883, 1.1396, 0.6908, 0.3546, 0.1584, 0.0658,
             0.0267, 0.0099, 0.0038, 0.0015, 0.0005, 0.0002, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        ], dtype=np.float32)
        self.crf_np = self.crf_np / np.max(np.sum(self.crf_np, axis=1))
        
        calib_path = os.path.join(os.path.dirname(root_dir), "calib.txt")
        if os.path.exists(calib_path):
            with open(calib_path, "r") as f:
                content = f.read().split()
                self.calib_vector = np.array([float(x) for x in content], dtype=np.float32)
        else:
            self.calib_vector = np.ones(31, dtype=np.float32)
            
        self.cubes_cache = []
        print(f"Pre-cargando {len(self.mat_files)} imágenes Harvard desde {os.path.basename(root_dir)}...")
        for mat_file in tqdm(self.mat_files, desc="Subiendo a RAM"):
            mat_data = scipy.io.loadmat(mat_file)
            cube = None
            for key in mat_data.keys():
                if not key.startswith('__') and isinstance(mat_data[key], np.ndarray) and mat_data[key].ndim == 3:
                    cube = mat_data[key].astype(np.float32)
                    break
            
            if cube is None:
                continue
                
            cube = cube * self.calib_vector.reshape(1, 1, 31)
            
            max_val = np.max(cube)
            if max_val > 0:
                cube = cube / max_val
                
            self.cubes_cache.append(cube)
            
        if not self.is_train and len(self.cubes_cache) > 0:
            H, W, C = self.cubes_cache[0].shape
            np.random.seed(42)
            self.mask_np = np.random.binomial(1, 0.5, (H, W)).astype(np.float32)
            self.mask_tensor = torch.from_numpy(self.mask_np).unsqueeze(0).unsqueeze(0)
            
        print("¡Precarga completada!")
        
    def _simulate_cassi(self, hsi_patch):
        H, W, C = hsi_patch.shape
        masked_patch = hsi_patch * self.mask_np[:, :, np.newaxis]
        cassi_meas = np.zeros((H, W + C - 1), dtype=np.float32)
        for i in range(C):
            cassi_meas[:, i:(i + W)] += masked_patch[:, :, i]
        return cassi_meas

    def _simulate_rgb(self, hsi_patch):
        H, W, C = hsi_patch.shape
        hsi_flat = hsi_patch.reshape(-1, C).T
        rgb_flat = np.dot(self.crf_np, hsi_flat)
        rgb_patch = rgb_flat.T.reshape(H, W, 3)
        return rgb_patch

    def __len__(self):
        if self.is_train:
            return len(self.mat_files) * self.num_patches
        else:
            return len(self.mat_files)

    def __getitem__(self, idx):
        if self.is_train:
            img_idx = idx // self.num_patches
            full_cube = self.cubes_cache[img_idx]
            
            max_h = full_cube.shape[0] - self.patch_size
            max_w = full_cube.shape[1] - self.patch_size
            
            rng = np.random.RandomState(idx) 
            start_h = rng.randint(0, max_h)
            start_w = rng.randint(0, max_w)
            patch_cube = full_cube[start_h : start_h + self.patch_size, start_w : start_w + self.patch_size, :]
        else:
            img_idx = idx
            patch_cube = self.cubes_cache[img_idx]
            
        patch_rgb = self._simulate_rgb(patch_cube)
        patch_cassi = self._simulate_cassi(patch_cube)
        
        tensor_cube = torch.from_numpy(patch_cube).permute(2, 0, 1)
        tensor_rgb = torch.from_numpy(patch_rgb).permute(2, 0, 1)
        tensor_cassi = torch.from_numpy(patch_cassi).unsqueeze(0)
        
        return tensor_cassi, tensor_rgb, tensor_cube

    def get_mask(self):
        return self.mask_tensor
