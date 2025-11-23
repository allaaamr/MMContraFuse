import torch
import os
import shap
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List
from scipy.ndimage import gaussian_filter, zoom
from matplotlib.colors import Normalize
import cv2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Encoder.mri_genomics import GenoMRI_Fusion
from models.Fusion.GatedTensorFusion import RadiomicMMF
class GenoMRIExplainer:
    """
    Explainability tool for GenoMRI_Fusion model.
    Provides attention visualization and Integrated Gradients analysis.
    """
    
    def __init__(self, 
                 model,
                 xai_dir: str,
                 train_loader,
                 val_loader,
                 train_split,
                 val_split,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: GenoMRI_Fusion model instance
            xai_dir: Directory to save XAI outputs
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            train_split: Training dataset split with genomic features
            val_split: Validation dataset split
            device: Device to run computations on
        """
        self.model = model.to(device)
        self.model.eval()
        self.xai_dir = xai_dir
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_split = train_split
        self.val_split = val_split
        self.device = device
        
        # Create output directories
        os.makedirs(xai_dir, exist_ok=True)
        os.makedirs(os.path.join(xai_dir, 'attention_maps'), exist_ok=True)
        os.makedirs(os.path.join(xai_dir, 'shap_plots'), exist_ok=True)
        
        # Storage for attention weights
        self.attention_weights = {}
        
    def run_all_explanations(self, target_case_id: str = "TCGA-02-0033"):
        """
        Run all explainability methods.
        
        Args:
            target_case_id: Case ID for attention visualization
        """
        print("=" * 80)
        print("Running GenoMRI Explainability Analysis")
        print("=" * 80)
        
        # 1. MRI Attention Visualization
        print("\n[1/2] Generating MRI attention maps...")
        self.plot_mri_attention(target_case_id)
        
        # 2. Genomics Integrated Gradients
        print("\n[2/2] Computing Integrated Gradients for genomics...")
        self.plot_genomic_attributions()
        
        print("\n" + "=" * 80)
        print("Explainability analysis complete!")
        print(f"Results saved to: {self.xai_dir}")
        print("=" * 80)
    
    # ==================== MRI Attention Visualization ====================
    
    def register_attention_hooks(self):
        """Register forward hooks to capture attention weights from MRI encoder."""
        self.attention_weights = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                self.attention_weights[name] = output.detach().cpu()
            return hook
        
        # Register hooks on MRI trunk's block1 and attention1
        hooks.append(self.model.block1.register_forward_hook(make_hook("Block1")))
        hooks.append(self.model.attention1.register_forward_hook(make_hook("Attention1")))
        
        return hooks
    
    def remove_hooks(self, hooks):
        """Remove all registered hooks."""
        for hook in hooks:
            hook.remove()
    
    def get_patient_by_case_id(self, loader, case_id: str):
        """
        Find a specific patient by case ID from the loader.
        
        Args:
            loader: DataLoader to search
            case_id: Target case ID (e.g., "TCGA-06-0213")
            
        Returns:
            Batch data for the patient or None if not found
        """
        for i, batch in enumerate(loader):
            # Assuming batch structure: [x_mri, x_path, x_omic, label, months, censored, slide_ids]
            slide_id = batch[6] if len(batch) > 6 else batch[-1]
            if slide_id[0][0] == case_id:
                # Return single patient data
                return batch
        
        return None
    
    def normalize_heatmap_mean(self, feature_map):
        """Normalize feature map by taking mean across channels."""
        if feature_map.dim() == 4:  # [B, C, H, W]
            heatmap = feature_map.mean(dim=1).squeeze()  # [H, W]
        elif feature_map.dim() == 3:  # [C, H, W]
            heatmap = feature_map.mean(dim=0)  # [H, W]
        else:
            heatmap = feature_map.squeeze()
        
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap.numpy()
    
    def normalize_heatmap_max(self, feature_map):
        """Normalize feature map by taking max across channels."""
        if feature_map.dim() == 4:  # [B, C, H, W]
            heatmap = feature_map.max(dim=1)[0].squeeze()  # [H, W]
        elif feature_map.dim() == 3:  # [C, H, W]
            heatmap = feature_map.max(dim=0)[0]  # [H, W]
        else:
            heatmap = feature_map.squeeze()
        
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap.numpy()
    
    def plot_mri_with_heatmaps(self, mri_slice, heatmap_before, heatmap_after, 
                               is_max_pool: bool, case_id: str):
        """
        Plot MRI slice with attention heatmaps overlaid.
        
        Args:
            mri_slice: Original MRI slice [H, W]
            heatmap_before: Heatmap before attention [H, W]
            heatmap_after: Heatmap after attention [H, W]
            is_max_pool: Whether max pooling was used for normalization
            case_id: Patient case ID
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original MRI slice
        axes[0].imshow(mri_slice.squeeze(), cmap='gray')
        axes[0].set_title('Original MRI Slice', fontsize=14)
        axes[0].axis('off')
        
        # Before attention
        axes[1].imshow(mri_slice.squeeze(), cmap='gray', alpha=0.7)
        im1 = axes[1].imshow(heatmap_before, cmap='jet', alpha=0.4)
        axes[1].set_title('Before Attention (Block1)', fontsize=14)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # After attention
        axes[2].imshow(mri_slice.squeeze(), cmap='gray', alpha=0.7)
        im2 = axes[2].imshow(heatmap_after, cmap='jet', alpha=0.4)
        axes[2].set_title('After Attention (Attention1)', fontsize=14)
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        
        plt.suptitle(f'MRI Attention Maps - {case_id}\n{"Max Pooling" if is_max_pool else "Mean Pooling"}', 
                     fontsize=16, y=1.02)
        
        pooling_type = "max" if is_max_pool else "mean"
        save_path = os.path.join(self.xai_dir, 'attention_maps', 
                                f'mri_attention_{case_id}_{pooling_type}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved attention map: {save_path}")
    
    def plot_32_slices(self, slices_tensor, case_id: str):
        """Plot all 32 MRI slices in a grid."""
        n_slices = slices_tensor.shape[0]
        n_cols = 8
        n_rows = (n_slices + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2.5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i in range(n_slices):
            axes[i].imshow(slices_tensor[i].squeeze(), cmap='gray')
            axes[i].set_title(f'Slice {i+1}', fontsize=10)
            axes[i].axis('off')
        
        # Hide remaining subplots
        for i in range(n_slices, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'All MRI Slices - {case_id}', fontsize=16)
        save_path = os.path.join(self.xai_dir, 'attention_maps', 
                                f'all_slices_{case_id}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved all slices: {save_path}")
    
    def plot_mri_attention(self, case_id: str = "TCGA-06-0213", slice_idx: int = 5):
        """
        Visualize attention maps for a specific patient's MRI.
        
        Args:
            case_id: Patient case ID
            slice_idx: Which slice to visualize (0-31)
        """
        print(f"\nSearching for patient {case_id}...")
        
        # Try to find patient in train loader first
        batch = self.get_patient_by_case_id(self.train_loader, case_id)
        if batch is None:
            print(f"Not found in training set, searching validation set...")
            batch = self.get_patient_by_case_id(self.val_loader, case_id)
        
        if batch is None:
            print(f"ERROR: Patient {case_id} not found in either dataset!")
            return
        
        print(f"Found patient {case_id}")
        
        # Extract data
        x_mri = batch[0].to(self.device)  # [1, S, H, W] or [1, S, C, H, W]
        x_omic = batch[2].to(self.device) if len(batch) > 2 else None
        
        # Register hooks
        hooks = self.register_attention_hooks()
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(x_mri=x_mri, x_omic=x_omic)
        
        # Remove hooks
        self.remove_hooks(hooks)
        
        # Extract attention weights
        if len(self.attention_weights) == 0:
            print("ERROR: No attention weights captured!")
            return
        
        print(f"Captured attention from {len(self.attention_weights)} layers")
        
        before_attn = self.attention_weights.get("Block1")
        after_attn = self.attention_weights.get("Attention1")
        
        if before_attn is None or after_attn is None:
            print("ERROR: Attention weights not properly captured!")
            return
        
        # Visualize all slices
        mri_slices = x_mri.cpu()[0]  # [S, H, W] or [S, C, H, W]
        if mri_slices.dim() == 4:  # [S, C, H, W]
            mri_slices = mri_slices[:, 0]  # Take first channel
        
        self.plot_32_slices(mri_slices, case_id)
        
        # Visualize specific slice with attention
        target_slice = mri_slices[slice_idx]
        
        # Normalize heatmaps (mean pooling)
        heatmap_before_mean = self.normalize_heatmap_mean(before_attn)
        heatmap_after_mean = self.normalize_heatmap_mean(after_attn)
        
        # Resize heatmaps to match MRI slice size
        target_size = target_slice.shape
        heatmap_before_mean = cv2.resize(heatmap_before_mean, 
                                         (target_size[1], target_size[0]))
        heatmap_after_mean = cv2.resize(heatmap_after_mean, 
                                        (target_size[1], target_size[0]))
        
        self.plot_mri_with_heatmaps(target_slice, heatmap_before_mean, 
                                    heatmap_after_mean, False, case_id)
        
        # Normalize heatmaps (max pooling)
        heatmap_before_max = self.normalize_heatmap_max(before_attn)
        heatmap_after_max = self.normalize_heatmap_max(after_attn)
        
        heatmap_before_max = cv2.resize(heatmap_before_max, 
                                        (target_size[1], target_size[0]))
        heatmap_after_max = cv2.resize(heatmap_after_max, 
                                       (target_size[1], target_size[0]))
        
        self.plot_mri_with_heatmaps(target_slice, heatmap_before_max, 
                                    heatmap_after_max, True, case_id)
        
        print(f"MRI attention visualization complete for {case_id}")
    
    # ==================== Genomics Integrated Gradients ====================
    
    def integrated_gradients(self, x_mri, x_omic, target_class: int, 
                            baseline=None, num_steps: int = 50):
        """
        Compute Integrated Gradients for genomic features.
        
        Args:
            x_mri: MRI input tensor
            x_omic: Genomic input tensor [1, n_features]
            target_class: Target class index
            baseline: Baseline for integration (default: zeros)
            num_steps: Number of interpolation steps
            
        Returns:
            Attribution scores for genomic features [n_features]
        """
        if baseline is None:
            baseline = torch.zeros_like(x_omic)
        
        baseline = baseline.to(self.device)
        x_omic = x_omic.to(self.device)
        x_mri = x_mri.to(self.device)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, num_steps + 1).to(self.device)
        interpolated_omics = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (x_omic - baseline)
            interpolated_omics.append(interpolated)
        
        interpolated_omics = torch.cat(interpolated_omics, dim=0)  # [num_steps+1, n_features]
        
        # Replicate MRI for all steps
        x_mri_repeated = x_mri.repeat(num_steps + 1, 1, 1, 1)
        
        # Compute gradients
        interpolated_omics.requires_grad_(True)
        
        logits = self.model(x_mri=x_mri_repeated, x_omic=interpolated_omics)
        target_logits = logits[:, target_class]
        
        # Compute gradients with respect to omic inputs
        gradients = torch.autograd.grad(
            outputs=target_logits.sum(),
            inputs=interpolated_omics,
            create_graph=False
        )[0]
        
        # Compute average gradients
        avg_gradients = gradients.mean(dim=0)  # [n_features]
        
        # Compute attributions: (x - baseline) * avg_gradients
        attributions = (x_omic.squeeze() - baseline.squeeze()) * avg_gradients
        
        return attributions.detach().cpu()
    
    def compute_global_attributions(self, loader, target_class: int = 0, 
                                   max_samples: Optional[int] = None):
        """
        Compute Integrated Gradients attributions across entire dataset.
        
        Args:
            loader: DataLoader to compute attributions for
            target_class: Target class index (0 for high risk)
            max_samples: Maximum number of samples to process (None for all)
            
        Returns:
            DataFrame with attributions for all samples
        """
        all_attributions = []
        n_processed = 0
        
        baseline = None
        
        for batch_idx, batch in enumerate(loader):
            x_mri = batch[0].to(self.device)
            x_omic = batch[2].to(self.device)
            
            if baseline is None:
                baseline = torch.zeros_like(x_omic[0:1])
            
            # Process each sample in batch
            for i in range(x_mri.shape[0]):
                attributions = self.integrated_gradients(
                    x_mri=x_mri[i:i+1],
                    x_omic=x_omic[i:i+1],
                    target_class=target_class,
                    baseline=baseline
                )
                all_attributions.append(attributions.numpy())
                
                n_processed += 1
                if max_samples is not None and n_processed >= max_samples:
                    break
            
            if max_samples is not None and n_processed >= max_samples:
                break
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {n_processed} samples...")
        
        print(f"Total samples processed: {n_processed}")
        
        # Stack all attributions
        all_attributions = np.vstack(all_attributions)
        
        # Get feature names
        feature_names = list(self.train_split.genomic_features.columns)
        
        # Create DataFrame
        attribution_df = pd.DataFrame(all_attributions, columns=feature_names)
        
        return attribution_df
    
    def plot_genomic_attributions(self, target_class: int = 3, top_k: int = 20):
        """
        Compute and visualize genomic feature attributions using SHAP-style plots.
        
        Args:
            target_class: Target class for attributions
            top_k: Number of top features to display
        """
        print(f"\nComputing global attributions for class {target_class}...")
        
        attribution_df = self.compute_global_attributions(
            self.train_loader, 
            target_class=target_class
        )
        
        # Save raw attributions
        csv_path = os.path.join(self.xai_dir, 'shap_plots', 
                               f'attributions_class{target_class}.csv')
        attribution_df.to_csv(csv_path, index=False)
        print(f"Saved attributions to {csv_path}")
        
        # Get top features by mean absolute attribution
        shap_values = attribution_df.values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_k:]
        
        top_features = [attribution_df.columns[i] for i in top_indices]
        top_shap_values = shap_values[:, top_indices]
        
        print(f"\nTop {top_k} most important features:")
        for i, (idx, feat) in enumerate(zip(top_indices[::-1], top_features[::-1])):
            print(f"  {i+1}. {feat}: {mean_abs_shap[idx]:.4f}")
        
        # Create SHAP beeswarm plot
        shap_explanation = shap.Explanation(
            values=top_shap_values,
            feature_names=top_features,
            data=top_shap_values
        )
        
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_explanation, show=False, max_display=top_k)
        
        save_path = os.path.join(self.xai_dir, 'shap_plots', 
                                f'genomic_beeswarm_class{target_class}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved beeswarm plot: {save_path}")
        
        # Create bar plot of mean absolute SHAP values
        plt.figure(figsize=(10, 6))
        sorted_indices = np.argsort(mean_abs_shap[top_indices])
        sorted_features = [top_features[i] for i in sorted_indices]
        sorted_values = mean_abs_shap[top_indices][sorted_indices]
        
        plt.barh(range(len(sorted_features)), sorted_values, color='steelblue')
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Mean |Attribution|', fontsize=12)
        plt.title(f'Top {top_k} Genomic Features by Attribution (Class {target_class})', 
                 fontsize=14)
        plt.tight_layout()
        
        save_path = os.path.join(self.xai_dir, 'shap_plots', 
                                f'genomic_importance_class{target_class}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved importance plot: {save_path}")



def create_explainer_from_checkpoint(checkpoint_path: str,
                                     args,
                                     train_loader,
                                     val_loader,
                                     train_split,
                                     val_split,
                                     xai_dir: str):
    """
    Convenience function to create explainer from a saved checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        args_path: Path to saved args pickle file (.pkl file)
        train_loader, val_loader: Data loaders
        train_split, val_split: Dataset splits
        xai_dir: Directory for XAI outputs
        
    Returns:
        GenoMRIExplainer instance
    """

        # Build model config from args
    model_config = {
        'omic_input_dim': args.omic_input_dim,
        'n_classes': args.n_classes,
    }

    optional_params = {
        'gate_path': False, 
        'gate_omic': False, 
        'scale_dim1': 8, 
        'scale_dim2':8, 
        'skip': False

    }
    
    for param, default_value in optional_params.items():
        model_config[param] = getattr(args, param, default_value)
    
    # Handle omic_hidden from model_size_omic
    # if hasattr(args, 'model_size_omic'):
    #     size_dict = {'small': [512, 512], 'big': [1024, 1024, 1024, 256]}
    #     model_config['omic_hidden'] = size_dict.get(args.model_size_omic, [512, 512])
    # else:
    #     model_config['omic_hidden'] = [512, 512]
    
    print(f"Model config: {model_config}")
    
    # Create model
    # model = GenoMRI_Fusion(**model_config)
    model = RadiomicMMF(**model_config)

            # Load weights
    print(f"Loading model weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # Create explainer
    explainer = GenoMRIExplainer(
        model=model,
        xai_dir=xai_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        train_split=train_split,
        val_split=val_split
    )
    
    print(f"Explainer created successfully!")
    return explainer

