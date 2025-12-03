import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import SweepEvalDataset, imagenet_transform
from model import NEJMbaseline


def infer_test(test_csv, model_path='checkpoints/best_model.pth',
               n_sweeps_test=8, output_csv='outputs/test_predictions.csv'):
    """
    Run inference on test set using a trained NEJMbaseline model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model = NEJMbaseline(pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepare test data
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    
    test_df = pd.read_csv(test_csv)
    test_dataset = SweepEvalDataset(
        test_csv, 
        n_sweeps=n_sweeps_test,
        transform=imagenet_transform, 
        target_frames=16
    )
    # test_loader = DataLoader(
    #     test_dataset, 
    #     batch_size=8,
    #     shuffle=False, 
    #     num_workers=4, 
    #     pin_memory=True
    # )

    test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=False
    )


    predictions = []
    study_ids = []
    sites = []

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing", leave=False)
        for i, (sweeps, _) in enumerate(test_pbar):
            sweeps = sweeps.to(device)                 # (B, S, T, C, H, W)
            B, S, T, C, H, W = sweeps.shape
            sweeps = sweeps.view(B * S, T, C, H, W)   # (B*S, T, C, H, W)

            outputs, _ = model(sweeps)                # (B*S, 1)
            outputs = outputs.view(B, S, 1).mean(dim=1)  # (B, 1) mean over sweeps
            preds = outputs.squeeze(1).cpu().numpy()

            predictions.extend(preds)

            start_idx = i * test_loader.batch_size
            end_idx = min(start_idx + B, len(test_df))
            study_ids.extend(test_df.iloc[start_idx:end_idx]['study_id'].tolist())
            sites.extend(test_df.iloc[start_idx:end_idx]['site'].tolist())

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Save predictions to CSV
    result_df = pd.DataFrame({
        'study_id': study_ids,
        'site': sites,
        'predicted_ga': predictions
    })
    result_df.to_csv(output_csv, index=False)
    print(f"✅ Saved predictions to {output_csv}")


if __name__ == "__main__":
    infer_test(
        test_csv='/mnt/Data/hackathon/final_test_split.csv',  # updated path
        model_path="checkpoints/best_model.pth",
        n_sweeps_test=8,
        output_csv="outputs/test_predictions.csv"
    )

