import os
import pickle
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from natsort import natsorted

from vip import load_vip


def load_snapshot(model, snapshot_path):
    """
    Loads finetuned weights into the given model (in DataParallel).
    Expects the snapshot dict to have a 'vip' key for model weights.
    """
    print(f"Loading finetuned weights from {snapshot_path}")
    snapshot = torch.load(snapshot_path, map_location="cpu")
    model.load_state_dict(snapshot["vip"], strict=True)
    print("Finished loading finetuned weights.")


def main():
    # 1) Initialize VIP model architecture (DataParallel)
    model = load_vip()  # Creates the VIP model in DataParallel
    model.eval()
    model.cuda()

    # 1b) Optional: Load a finetuned snapshot
    snapshot_path = "/home/emlyn/rl_franka/vip/vip/vipoutput/train_vip_finetune/2025-03-21_12-17-39/snapshot_8500.pt"
    load_snapshot(model, snapshot_path)
    model.eval()  # ensure eval mode

    # 2) Define transform for each image
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor()
    ])

    # Path to your dataset
    data_dir = Path("/home/emlyn/datasets/strawb_sim/success/train/robot")

    # List to store all final-frame embeddings
    all_final_embeddings = []

    # 3) Loop over each “video” subfolder
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        # Gather all frames in this subfolder
        frames = natsorted(subdir.glob("*.png"))
        if len(frames) == 0:
            # No PNG files here, skip
            continue

        # 4) Load the final frame in this folder
        final_frame_path = frames[-1]
        img = Image.open(final_frame_path).convert("RGB")
        # opencv_image = np.array(img)
        # cv2.imshow("image", opencv_image)
        # cv2.waitKey(0)

        img_t = transform(img)            # shape: [3, 224, 224]
        img_t = img_t.unsqueeze(0).cuda() # shape: [1, 3, 224, 224]
        
        # 5) Forward pass through VIP (finetuned)
        with torch.no_grad():
            # VIP typically expects [0..255] scale
            vip_input = img_t * 255
            emb = model(vip_input)        # shape: [1, embedding_dim]

        # Convert to numpy & store
        emb_np = emb.squeeze(0).cpu().numpy()
        all_final_embeddings.append(emb_np)

    # 6) Compute mean embedding across all videos
    if len(all_final_embeddings) > 0:
        all_final_embeddings = np.stack(all_final_embeddings, axis=0)
        mean_embedding = all_final_embeddings.mean(axis=0)
    else:
        mean_embedding = None  # In case no valid frames found

    # 7) Save mean embedding as pkl
    out_path = "mean_embedding.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(mean_embedding, f)

    print(f"Saved mean embedding to {out_path}")


if __name__ == "__main__":
    main()
