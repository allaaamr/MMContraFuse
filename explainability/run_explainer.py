
import pickle
import torch
import os
import argparse
from explainer import create_explainer_from_checkpoint

def pickle_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def unpickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


parser = argparse.ArgumentParser(description='XAI Analysis for GenoMRI_Fusion on TCGA Data.')

# Required paths
parser.add_argument('--checkpoint', type=str, 
                    default='/home/alaa.mohamed/MMContraFuse/results/genomic_gtf_risk/best_model_radiomic.pt',
                    help='Path to model checkpoint (.pt file)')
parser.add_argument('--args', type=str, 
                    default='/home/alaa.mohamed/MMContraFuse/results/genomic_gtf_risk/args_radiomic.pkl',
                    help='Path to saved args (.pkl file)')
parser.add_argument('--xai_dir', type=str, 
                    default='/home/alaa.mohamed/MMContraFuse/results/xai_results/radiogenomic_gtf_risk',
                    help='Directory for XAI outputs')
parser.add_argument('--data', type=str, 
                    default='/home/alaa.mohamed/MMContraFuse/data/2.5D_MRIs',
                    help='Path to data directory')
parser.add_argument('--val_loader', type=str, 
                    default='/home/alaa.mohamed/MMContraFuse/results/genomic_gtf_risk/val_loader_radiomic.pkl',
                    help='Path to val loader')
parser.add_argument('--train_loader', type=str, 
                default='/home/alaa.mohamed/MMContraFuse/results/genomic_gtf_risk/train_loader_radiomic.pkl',
                help='Path to train loader')
parser.add_argument('--val_split', type=str, 
                    default='/home/alaa.mohamed/MMContraFuse/results/genomic_gtf_risk/val_split_radiomic.pkl',
                    help='Path to val split')
parser.add_argument('--train_split', type=str, 
                    default='/home/alaa.mohamed/MMContraFuse/results/genomic_gtf_risk/train_split_radiomic.pkl',
                    help='Path to train split')

# XAI parameters
parser.add_argument('--case_id', type=str, default='TCGA-02-0047',
                    help='Target case ID for MRI attention visualization')
parser.add_argument('--slice_idx', type=int, default=5,
                    help='MRI slice index to visualize (0-31)')
parser.add_argument('--target_class', type=int, default=0,
                    help='Target class for genomic attributions (0=high risk)')
parser.add_argument('--top_k', type=int, default=10,
                    help='Number of top genomic features to display')

# Analysis options
parser.add_argument('--skip_mri', action='store_true',
                    help='Skip MRI attention visualization')
parser.add_argument('--skip_genomic', action='store_true',
                    help='Skip genomic attribution analysis')
parser.add_argument('--max_samples', type=int, default=None,
                    help='Maximum samples for genomic attributions (None=all)')
args = parser.parse_args()

def main(args):

    # Create output directory
    os.makedirs(args.xai_dir, exist_ok=True)
    
    # # ========== Load pickled objects (data loaders and args)  ==========
    val_loader = unpickle(args.val_loader)
    val_split = unpickle( args.val_split)		
    train_loader = unpickle(args.train_loader)
    train_split = unpickle(args.train_split )
    args_loaded = unpickle(args.args)


    # # ========== CREATE EXPLAINER ==========
    print(" Creating explainer...")
    explainer = create_explainer_from_checkpoint(
        checkpoint_path=args.checkpoint,
        args=args_loaded,
        train_loader=train_loader,
        val_loader=val_loader,
        train_split=train_split,
        val_split=val_split,
        xai_dir=args.xai_dir
    )
    
    # ========== RUN MRI ATTENTION ANALYSIS ==========
    if not args.skip_mri:
        print("\n[Step 3/4] Running MRI attention visualization...")
        print(f"Target patient: {args.case_id}")
        print(f"Target slice: {args.slice_idx}")
        
        explainer.plot_mri_attention(
            case_id=args.case_id,
            slice_idx=args.slice_idx
        )
    else:
        print("\n[Step 3/4] Skipping MRI attention visualization (--skip_mri)")
    
    # # ========== RUN GENOMIC ATTRIBUTION ANALYSIS ==========
    if not args.skip_genomic:
        print("\n[Step 4/4] Running genomic attribution analysis...")
        print(f"Target class: {args.target_class}")
        print(f"Top features: {args.top_k}")
        if args.max_samples:
            print(f"Max samples: {args.max_samples}")
        
        # Optionally limit samples for faster computation
        if args.max_samples:
            original_method = explainer.compute_global_attributions
            def limited_compute(*args_func, **kwargs):
                kwargs['max_samples'] = args.max_samples
                return original_method(*args_func, **kwargs)
            explainer.compute_global_attributions = limited_compute
        
        explainer.plot_genomic_attributions(
            target_class=args.target_class,
            top_k=args.top_k
        )
    else:
        print("\n[Step 4/4] Skipping genomic attribution analysis (--skip_genomic)")
    
    # # ========== SUMMARY ==========
    # print("\n" + "=" * 80)
    # print("XAI Analysis Complete!")
    # print("=" * 80)
    # print(f"Results saved to: {args.xai_dir}")
    # print("\nGenerated outputs:")
    # if not args.skip_mri:
    #     print(f"  MRI attention maps: {args.xai_dir}/attention_maps/")
    # if not args.skip_genomic:
    #     print(f"  Genomic SHAP plots: {args.xai_dir}/shap_plots/")
    # print("=" * 80)


if __name__ == "__main__":
    main(args)