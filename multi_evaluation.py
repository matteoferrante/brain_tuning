

import torch
import wandb
import numpy as np
import tqdm
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


## EVALUATION FUNCTIONS

def organize_multi_results(loader,multi_model):
    # Initialize the dictionary to store results for each subject
    results = defaultdict(lambda: {"y_pred": [], "y_gt": [], "gt_images": [], "gt_texts": []})

    # Run inference and store predictions and ground truth
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):

            k = batch["subject_id"]
            img = batch["images"]
            captions = batch["captions"]


            y_hat_fmri, y_hat_embeddings = multi_model(batch)
            for subject_index in torch.unique(k):
                mask = k == subject_index
                results[subject_index.item()]["y_pred"].append(y_hat_fmri[mask])
                results[subject_index.item()]["y_gt"].append(y_hat_embeddings[mask])
                results[subject_index.item()]["gt_images"].extend(img[mask])
                results[subject_index.item()]["gt_texts"].extend([captions[idx] for idx in torch.where(mask)[0]])


    # Concatenate the results for each subject
    for subject_index in results:
        results[subject_index]["y_pred"] = torch.cat(results[subject_index]["y_pred"], dim=0)
        results[subject_index]["y_gt"] = torch.cat(results[subject_index]["y_gt"], dim=0)
    
    return results

# Identification accuracy function
def identification_accuracy_fast(P, T):
    n = P.shape[0]
    if n == 0:
        return np.nan

    P_mean = np.mean(P, axis=1, keepdims=True)
    T_mean = np.mean(T, axis=1, keepdims=True)
    P_std = np.std(P, axis=1, keepdims=True)
    T_std = np.std(T, axis=1, keepdims=True)
    
    P_normalized = (P - P_mean) / P_std
    T_normalized = (T - T_mean) / T_std
    C = np.dot(P_normalized, T_normalized.T) / P.shape[1]

    id_acc = np.zeros(n)
    for i in range(n):
        id_acc[i] = (C[i, i] >= C[i]).sum()  
        id_acc[i] = id_acc[i] / (n - 1)
    
    return np.mean(id_acc)


def evaluate_and_log(loader, multi_model):
    results = organize_multi_results(loader, multi_model)
    
    # Initialize lists to store results for each subject
    subject_ids, top1_accuracies, top5_accuracies, id_accuracies, id_accuracy_baseline = [], [], [], [], []
    top1_baseline, top5_baseline = [], []
    top1_improvement_over, top5_improvement_over = [], []
    similarity_matrices = {}
    
    print("Starting evaluation...")
    for subject_index, subject_data in results.items():
        print(f"Evaluating metrics for subject {subject_index}...")
        
        y_pred = subject_data["y_pred"]
        y_gt = subject_data["y_gt"]
        gt_images = subject_data["gt_images"]

        y_pred_np = y_pred.cpu().numpy()
        y_gt_np = y_gt.cpu().numpy()

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(y_pred_np, y_gt_np)
        similarity_matrices[subject_index] = similarity_matrix  # Store similarity matrix
        print(f"Computed similarity matrix for subject {subject_index}.")

        # Calculate Top-1 and Top-5 accuracy
        top1_acc = (np.argmax(similarity_matrix, axis=1) == np.arange(len(y_gt_np))).mean()
        top5_acc = np.mean([1 if i in np.argsort(-similarity_matrix[i])[:5] else 0 for i in range(len(y_gt_np))])
        print(f"Top-1 Accuracy: {top1_acc:.4f}, Top-5 Accuracy: {top5_acc:.4f} for subject {subject_index}.")

        # Identification accuracy
        id_acc = identification_accuracy_fast(y_pred_np, y_gt_np)
        print(f"Identification accuracy for subject {subject_index}: {id_acc:.4f}")

        # Baseline performance (50% for identification accuracy as requested)
        id_acc_baseline = 50

        # Log metrics for this subject
        wandb.log({
            f"subject_{subject_index}_top1_acc": top1_acc,
            f"subject_{subject_index}_top5_acc": top5_acc,
            f"subject_{subject_index}_identification_accuracy": id_acc,
        })

        # Store results in lists
        subject_ids.append(subject_index)
        top1_accuracies.append(100*top1_acc)
        top5_accuracies.append(100*top5_acc)
        id_accuracies.append(100*id_acc)
        id_accuracy_baseline.append(id_acc_baseline)
        top1_baseline.append(100/similarity_matrix.shape[0])
        top5_baseline.append(500/similarity_matrix.shape[0])

        top1_improvement_over.append(top1_acc/(1/similarity_matrix.shape[0]))
        top5_improvement_over.append(top5_acc/(5/similarity_matrix.shape[0]))


        # Log a sample of 10 images and top-5 retrievals for each subject
        sample_indices = np.random.choice(len(y_gt_np), 10, replace=False)
        subject_table = wandb.Table(columns=["Stimulus Image", "Top 5 Retrieved Images"])

        for i in sample_indices:
            sim_scores = similarity_matrix[i]
            top5_indices = np.argsort(-sim_scores)[:5]
            top5_images = [wandb.Image(gt_images[idx]) for idx in top5_indices]  # Convert to wandb.Image
            stimulus_image = wandb.Image(gt_images[i])  # Convert stimulus to wandb.Image
            subject_table.add_data(stimulus_image, top5_images)
        
        print(f"Logged top-5 retrievals for subject {subject_index}.")
        
        # Log the table for each subject
        wandb.log({f"subject_{subject_index}_Top5_Retrievals": subject_table})

    print("Evaluation complete. Results loaded to wandb.")
    
    # Create a DataFrame to store overall results per subject
    results_df = pd.DataFrame({
        "Subject": subject_ids,
        "Identification Accuracy (%)": id_accuracies,
        "ID Accuracy Baseline (%)": id_accuracy_baseline,
        "Top-1 Accuracy (%)": top1_accuracies,
        "Top1 Baseline (%)": top1_baseline,
        "Top1 Improvement Over Baseline": top1_improvement_over,
        "Top-5 Accuracy (%)": top5_accuracies,
        "Top5 Baseline (%)": top5_baseline,
        "Top5 Improvement Over Baseline": top5_improvement_over
    })

    # Return the results dictionary and similarity matrices for further analysis or visualization
    return results_df, similarity_matrices, results



