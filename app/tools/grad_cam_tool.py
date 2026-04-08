"""
grad_cam_tool.py
────────────────────────────────────────────────────────────────────────────
Grad-CAM heatmap generator for 3D MRI brain scans.
Includes 3D visualisations: NIfTI, Animated GIF, and Interactive Plotly HTML.
Corrected Version: Fixes PyTorch 'leaf variable inplace' and MONAI MetaTensor bugs.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityd,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Hook-based Grad-CAM (Tensor-Hook Approach)
# ──────────────────────────────────────────────────────────────────────────────

class GradCAM3D:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        self._activations = output
        
        def _tensor_backward_hook(grad):
            self._gradients = grad
            
        self._activations.register_hook(_tensor_backward_hook)

    def remove_hooks(self):
        self._fwd_handle.remove()

    def generate(self, input_tensor: torch.Tensor, target_channel: int = 0) -> np.ndarray:
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        score = output[:, target_channel, ...].sum()
        
        # Backward pass
        score.backward(retain_graph=False)

        # Calcul de la carte Grad-CAM
        weights = self._gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="trilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
            
        return cam

# ──────────────────────────────────────────────────────────────────────────────
# 2. Helpers pour la résolution de couche et la visualisation
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_target_layer(model: SegResNet) -> torch.nn.Module:
    try:
        last_block = list(model.down_layers)[-1]
        for name, m in reversed(list(last_block.named_modules())):
            if isinstance(m, torch.nn.Conv3d):
                return m
    except Exception:
        pass
    for name, m in reversed(list(model.named_modules())):
        if isinstance(m, torch.nn.Conv3d):
            return m
    raise RuntimeError("Could not find a Conv3d layer.")

def _overlay_heatmap_on_slice(mri_slice, cam_slice, alpha=0.45, colormap="jet"):
    s_min, s_max = mri_slice.min(), mri_slice.max()
    mri_norm = (mri_slice - s_min) / (s_max - s_min) if (s_max - s_min) > 1e-8 else mri_slice
    mri_rgb = np.stack([mri_norm] * 3, axis=-1)
    cmap = plt.get_cmap(colormap)
    heatmap_rgb = cmap(cam_slice)[..., :3]
    overlay = (1 - alpha) * mri_rgb + alpha * heatmap_rgb
    return np.clip(overlay * 255, 0, 255).astype(np.uint8)

def _save_slice_grid(mri_vol, cam_vol, output_path, plane="axial", n_slices=6, alpha=0.45, colormap="jet"):
    D, H, W = mri_vol.shape
    if plane == "axial":
        indices = np.linspace(D * 0.15, D * 0.85, n_slices, dtype=int)
        get_slices = lambda i: (mri_vol[i], cam_vol[i])
    elif plane == "coronal":
        indices = np.linspace(H * 0.15, H * 0.85, n_slices, dtype=int)
        get_slices = lambda i: (mri_vol[:, i, :], cam_vol[:, i, :])
    else:
        indices = np.linspace(W * 0.15, W * 0.85, n_slices, dtype=int)
        get_slices = lambda i: (mri_vol[:, :, i], cam_vol[:, :, i])

    fig, axes = plt.subplots(2, n_slices, figsize=(n_slices * 3, 7))
    fig.patch.set_facecolor("#0d0d0d")

    for col, idx in enumerate(indices):
        mri_s, cam_s = get_slices(idx)
        axes[0, col].imshow(mri_s.T, cmap="gray", origin="lower", aspect="auto")
        axes[0, col].axis("off")
        overlay = _overlay_heatmap_on_slice(mri_s, cam_s, alpha=alpha, colormap=colormap)
        axes[1, col].imshow(overlay.transpose(1, 0, 2), origin="lower", aspect="auto")
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 3. GradCAMTool — L'outil principal pour l'Agent
# ──────────────────────────────────────────────────────────────────────────────

class GradCAMTool:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        spatial_size: tuple = (128, 128, 128),
        output_dir: str = "reports/gradcam",
        target_channel: int = 1,
        alpha: float = 0.45,
        colormap: str = "inferno",
        save_nifti: bool = True,
        save_gif: bool = True,
        save_html_3d: bool = True,
    ):
        self.spatial_size = spatial_size
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_channel = target_channel
        self.alpha = alpha
        self.colormap = colormap
        self.save_nifti = save_nifti
        self.save_gif = save_gif
        self.save_html_3d = save_html_3d
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SegResNet(
            blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1],
            init_filters=16, in_channels=4, out_channels=3,
        ).to(self.device)

        if weights_path and Path(weights_path).exists():
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

        for m in self.model.modules():
            if hasattr(m, "inplace"):
                m.inplace = False

        self.model.eval()
        self._gradcam = GradCAM3D(self.model, _resolve_target_layer(self.model))

        self._transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=self.spatial_size),
        ])

    def execute(self, input_source, patient_name="patient", planes=("axial", "coronal", "sagittal"), n_slices=6):
        logger.info(f"🔬 GradCAMTool: Processing [{patient_name}] -> {self.output_dir}")
        
        mri_np, input_tensor = self._load_and_preprocess(input_source)
        input_tensor = input_tensor.to(self.device)

        with torch.enable_grad():
            cam_vol = self._gradcam.generate(input_tensor, target_channel=self.target_channel)

        output_files = []
        
        if self.save_nifti:
            nii_path = str(self.output_dir / f"{patient_name}_gradcam.nii.gz")
            nib.save(nib.Nifti1Image(cam_vol.astype(np.float32), np.eye(4)), nii_path)
            output_files.append(nii_path)

        for plane in planes:
            png_path = str(self.output_dir / f"{patient_name}_gradcam_{plane}.png")
            _save_slice_grid(mri_np, cam_vol, png_path, plane=plane, n_slices=n_slices, alpha=self.alpha, colormap=self.colormap)
            output_files.append(png_path)

        if self.save_gif:
            gif_path = str(self.output_dir / f"{patient_name}_3D_scan.gif")
            self._save_animated_gif(mri_np, cam_vol, gif_path)
            output_files.append(gif_path)

        if self.save_html_3d:
            html_path = str(self.output_dir / f"{patient_name}_3D_model.html")
            self._save_plotly_3d(mri_np, cam_vol, html_path) # Ajout de mri_np ici !
            output_files.append(html_path)

        return {"cam_volume": cam_vol, "output_files": output_files}

    def _load_and_preprocess(self, input_source) -> tuple[np.ndarray, torch.Tensor]:
        data = {"image": str(input_source)}
        transformed = self._transforms(data)
        
        tensor = transformed["image"]
        # CORRECTIF : Retirer le MetaTensor de MONAI pour éviter les bugs avec l'autograd
        if hasattr(tensor, "as_tensor"):
            tensor = tensor.as_tensor()
            
        tensor = tensor.unsqueeze(0).float()
        mri_np = transformed["image"][0].numpy()

        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 4, 1, 1, 1)

        # CORRECTIF : Pas de requires_grad_(True) ici pour éviter l'erreur "leaf variable inplace"
        return mri_np, tensor

    # --- FONCTIONS 3D ---
    def _save_animated_gif(self, mri_vol, cam_vol, output_path):
        try:
            from PIL import Image
            frames = []
            for i in range(mri_vol.shape[0]):
                mri_s = mri_vol[i]
                cam_s = cam_vol[i]
                if mri_s.max() < 0.05: continue
                
                overlay = _overlay_heatmap_on_slice(mri_s, cam_s, self.alpha, self.colormap)
                img = Image.fromarray(overlay).transpose(Image.ROTATE_90)
                frames.append(img)
            
            if frames:
                frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
                logger.info(f"🎞️ GIF 3D sauvegardé : {output_path}")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération du GIF : {e}")

    def _save_plotly_3d(self, mri_vol, cam_vol, output_path):
        try:
            import plotly.graph_objects as go
            from scipy.ndimage import zoom
            import numpy as np
            
            # Redimensionnement (zoom) pour que le navigateur ne plante pas
            zoom_factor = 64 / max(cam_vol.shape)
            
            cam_down = zoom(cam_vol, zoom_factor)
            mri_down = zoom(mri_vol, zoom_factor)
            
            # Normalisation stricte entre 0 et 1 pour les deux volumes
            mri_down = (mri_down - mri_down.min()) / (mri_down.max() - mri_down.min() + 1e-8)
            cam_down = (cam_down - cam_down.min()) / (cam_down.max() - cam_down.min() + 1e-8)
            
            # 🔪 TUEUR DE CUBE : On force tout ce qui n'est pas le cerveau ou la tumeur à ZÉRO absolu
            mri_down[mri_down < 0.15] = 0.0  # Supprime le cube gris (l'air autour de la tête)
            cam_down[cam_down < 0.60] = 0.0  # Ne garde que le cœur de la tumeur (les 40% les plus intenses)
            
            X, Y, Z = np.mgrid[0:cam_down.shape[0], 0:cam_down.shape[1], 0:cam_down.shape[2]]
            
            # 🧠 COUCHE 1 : Le Cerveau (Fantôme transparent)
            brain_trace = go.Volume(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=mri_down.flatten(),
                isomin=0.01,       # Affiche tout ce qui a survécu au filtre au-dessus
                isomax=1.0,
                opacity=0.15,      # Effet fantôme/verre
                surface_count=15,
                colorscale='Greys',
                showscale=False,
                hoverinfo='skip',
                caps=dict(x_show=False, y_show=False, z_show=False) # 🚫 INTERDIT À PLOTLY DE DESSINER LES BORDS DU CUBE
            )
            
            # 🔴 COUCHE 2 : La Tumeur en Rouge
            tumor_trace = go.Volume(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=cam_down.flatten(),
                isomin=0.61,       # Seuil d'affichage de la tumeur
                isomax=1.0,
                opacity=0.9,       # Tumeur très visible (solide)
                surface_count=15,
                colorscale='Reds', # Couleur ROUGE pure
                name='Tumeur',
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False) # 🚫 INTERDIT À PLOTLY DE DESSINER LES BORDS DU CUBE
            )
            
            fig = go.Figure(data=[brain_trace, tumor_trace])
            
            # Nettoyage absolu de la scène (fond noir, aucune ligne, aucune grille)
            fig.update_layout(
                scene=dict(
                    xaxis=dict(visible=False, showbackground=False),
                    yaxis=dict(visible=False, showbackground=False),
                    zaxis=dict(visible=False, showbackground=False),
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                paper_bgcolor='#000000', # Fond noir absolu pour faire ressortir le rouge et le gris
                plot_bgcolor='#000000',
                showlegend=False
            )
            
            fig.write_html(output_path)
            logger.info(f"🌐 Modèle 3D HTML (Cerveau + Tumeur Rouge) sauvegardé : {output_path}")
        except ImportError:
            logger.warning("⚠️ 'plotly' non installé. Ignorer la génération HTML 3D.")
        except Exception as e:
            logger.error(f"❌ Erreur génération HTML 3D : {e}")