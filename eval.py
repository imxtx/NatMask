import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from skimage.io import imread
from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd
from models import FACE_MODELS
from utils.image import split_image


def compute_similarity(device, model, adv_imgs, tar_imgs):
    """Compute cosine similarity between adversarial and target images
    Args:
        device: torch device
        model: face recognition model dict with 'net' and 'transform'
        adv_imgs: (N, H, W, C) adversarial images
        tar_imgs: (N, H, W, C) target images
    Returns:
        np.ndarray: similarities for each pair
    """
    # Transform images
    adv_imgs_tensor = model["transform"](adv_imgs).to(device)
    tar_imgs_tensor = model["transform"](tar_imgs).to(device)

    # Extract features
    with torch.no_grad():
        adv_feats = model["net"](adv_imgs_tensor)
        tar_feats = model["net"](tar_imgs_tensor)

    # Compute cosine similarity
    similarities = F.cosine_similarity(adv_feats, tar_feats, dim=1)
    return similarities.cpu().numpy()


def evaluate_model_attack(device, model_name, attack_type, results_dir, batch_size=32):
    """Evaluate a specific model on a specific attack type
    Args:
        device: torch device
        model_name: name of the face recognition model
        attack_type: 'dodging' or 'impersonation'
        results_dir: path to results directory
        batch_size: batch size for evaluation
    Returns:
        dict: evaluation results
    """
    # Load model
    model = FACE_MODELS[model_name]
    model["net"] = model["loader"](device)
    model["net"].to(device)
    model["net"].eval()

    # Path to attack results
    attack_dir = Path(results_dir) / model_name / attack_type

    if not attack_dir.exists():
        print(f"Warning: Directory {attack_dir} does not exist, skipping...")
        return {"asr": 0.0, "avg_similarity": 0.0, "total_samples": 0}

    all_similarities = []
    total_samples = 0

    # Process each batch directory
    batch_dirs = sorted(
        [d for d in attack_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")]
    )

    for batch_dir in tqdm(batch_dirs, desc=f"Evaluating {model_name}/{attack_type}"):
        images_dir = batch_dir / "images"
        if not images_dir.exists():
            continue

        # Get all image files
        image_files = sorted(list(images_dir.glob("*.png")))
        if not image_files:
            continue

        # Process images in batches
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i : i + batch_size]

            # Load and stack images
            batch_images = []
            for img_file in batch_files:
                img = imread(str(img_file))
                if img.shape[-1] == 4:  # Remove alpha channel if present
                    img = img[:, :, :3]
                batch_images.append(img)

            if not batch_images:
                continue

            batch_images = np.stack(batch_images)  # (N, H, W, C)

            # Split into adversarial and target parts
            adv_imgs, tar_imgs = split_image(batch_images)

            # Compute similarities
            similarities = compute_similarity(device, model, adv_imgs, tar_imgs)
            all_similarities.extend(similarities)
            total_samples += len(similarities)

    if total_samples == 0:
        return {"asr": 0.0, "avg_similarity": 0.0, "total_samples": 0}

    all_similarities = np.array(all_similarities)
    avg_similarity = all_similarities.mean()

    # Calculate Attack Success Rate (ASR)
    threshold = model["threshold"]
    if attack_type == "dodging":
        # For dodging: success when similarity < threshold (low similarity)
        asr = (all_similarities < threshold).mean() * 100
    elif attack_type == "impersonation":
        # For impersonation: success when similarity >= threshold (high similarity)
        asr = (all_similarities >= threshold).mean() * 100
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    return {
        "asr": asr,
        "avg_similarity": avg_similarity,
        "total_samples": total_samples,
        "similarities": all_similarities,
    }


def save_results(results, save_path):
    """Save evaluation results to files
    Args:
        results: evaluation results dictionary
        save_path: path to save results
    """
    os.makedirs(save_path, exist_ok=True)

    # Save summary table
    summary_data = []
    for model_name in results:
        for attack_type in results[model_name]:
            result = results[model_name][attack_type]
            summary_data.append(
                {
                    "Model": model_name,
                    "Attack": attack_type,
                    "ASR (%)": f"{result['asr']:.2f}",
                    "Avg Similarity": f"{result['avg_similarity']:.4f}",
                    "Total Samples": result["total_samples"],
                }
            )

    # Create DataFrame and save
    df = pd.DataFrame(summary_data)
    summary_file = os.path.join(save_path, "evaluation_summary.csv")
    df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

    # Print results table
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(
        f"{'Model':<12} {'Attack':<14} {'ASR (%)':<10} {'Avg Sim':<10} {'Samples':<10}"
    )
    print("-" * 80)

    for model_name in ["facenet", "cosface", "arcface", "mobileface"]:
        if model_name in results:
            for attack_type in ["dodging", "impersonation"]:
                if attack_type in results[model_name]:
                    result = results[model_name][attack_type]
                    print(
                        f"{model_name:<12} {attack_type:<14} {result['asr']:<10.2f} "
                        f"{result['avg_similarity']:<10.4f} {result['total_samples']:<10}"
                    )
    print("=" * 80)

    # Save detailed results for each model-attack combination
    for model_name in results:
        for attack_type in results[model_name]:
            result = results[model_name][attack_type]
            if result["total_samples"] > 0:
                detail_file = os.path.join(
                    save_path, f"{model_name}_{attack_type}_details.txt"
                )
                with open(detail_file, "w") as f:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Attack: {attack_type}\n")
                    f.write(f"ASR: {result['asr']:.2f}%\n")
                    f.write(f"Average Similarity: {result['avg_similarity']:.4f}\n")
                    f.write(f"Total Samples: {result['total_samples']}\n")
                    f.write(f"Threshold: {FACE_MODELS[model_name]['threshold']:.4f}\n")
                    if "similarities" in result:
                        sims = result["similarities"]
                        f.write(f"Similarity Stats:\n")
                        f.write(f"  Min: {sims.min():.4f}\n")
                        f.write(f"  Max: {sims.max():.4f}\n")
                        f.write(f"  Std: {sims.std():.4f}\n")


def main(args):
    """Main evaluation function"""
    device = torch.device(args.device)
    models = ["facenet", "cosface", "arcface", "mobileface"]
    attacks = ["dodging", "impersonation"]

    print(f"Starting evaluation on device: {device}")
    print(f"Results directory: {args.results_dir}")
    print(f"Batch size: {args.batch_size}")

    # Evaluate all combinations
    results = {}
    for model_name in models:
        print(f"\n[Evaluating model: {model_name}]")
        results[model_name] = {}

        for attack_type in attacks:
            print(f"  Attack type: {attack_type}")
            try:
                result = evaluate_model_attack(
                    device=device,
                    model_name=model_name,
                    attack_type=attack_type,
                    results_dir=args.results_dir,
                    batch_size=args.batch_size,
                )
                results[model_name][attack_type] = result
                print(
                    f"    ASR: {result['asr']:.2f}% | "
                    f"Avg Sim: {result['avg_similarity']:.4f} | "
                    f"Samples: {result['total_samples']}"
                )
            except Exception as e:
                print(f"    Error: {e}")
                results[model_name][attack_type] = {
                    "asr": 0.0,
                    "avg_similarity": 0.0,
                    "total_samples": 0,
                }

    # Save results
    save_results(results, args.save_dir)
    print(f"\nEvaluation completed! Results saved to: {args.save_dir}")


def parse_args():
    parser = ArgumentParser(
        description="Evaluate face recognition attack success rates"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run evaluation on (default: cuda:0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/lfw",
        help="Path to results directory (default: results/lfw)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/lfw/eval",
        help="Directory to save evaluation results (default: results/lfw/eval)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Evaluation Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    main(args)
