import cv2
import glob
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import numpy as np
import os
import pickle

import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

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

def load_embedding(rep='vip'):
    """
    Loads the model and transform for VIP, R3M, or ResNet embeddings.
    """
    if rep == "vip":
        model = load_vip()
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()
        ])
    elif rep == "r3m":
        from r3m import load_r3m
        model = load_r3m("resnet50")
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()
        ])
    elif rep == "resnet":
        model = models.resnet50(pretrained=True, progress=False)
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])
    return model, transform

def get_video_frames(filepath):
    """
    Loads video frames from a given filepath (OpenCV),
    returns a list of frames in RGB order.
    """
    vidcap = cv2.VideoCapture(filepath)
    frames = []
    while True:
        success, image = vidcap.read()
        if not success:
            break
        # Convert BGR (OpenCV) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
    return frames

def get_embeddings(frames, model, transform, rep='vip'):
    """
    Convert frames to embeddings.
    """
    # Transform frames
    imgs_cur = [transform(Image.fromarray(f.astype(np.uint8))) for f in frames]

    # Stack into a single tensor
    imgs_cur = torch.stack(imgs_cur)

    # For VIP or R3M, multiply by 255
    if rep in ['vip', 'r3m']:
        imgs_cur = imgs_cur * 255

    # Optionally truncate for very long videos
    imgs_cur = imgs_cur[:200]

    # Get embeddings
    with torch.no_grad():
        embeddings = model(imgs_cur.cuda()).cpu().numpy()

    return embeddings, imgs_cur

def animate_distance(
    distances, frames_tensor, goal_frame_tensor, outpath, rep='vip', 
    embedding_name='VIP', line_color='tab:blue'
):
    """
    Creates and saves a GIF animation of the embedding distance over time,
    and a side-by-side frame from the video at each step.
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # First, we do a static plot to get axis limits
    ax[0].plot(np.arange(len(distances)), distances, color=line_color, linewidth=3)
    ax[1].imshow(goal_frame_tensor.permute(1, 2, 0) / 255)

    ax[0].set_title(f"{embedding_name} Embedding Distance", fontsize=15)
    ax[0].set_xlabel("Frame", fontsize=15)
    ax[0].set_ylabel("Distance", fontsize=15)
    ax[1].set_title("Goal Frame", fontsize=15)
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax0_xlim = ax[0].get_xlim()
    ax0_ylim = ax[0].get_ylim()
    ax1_xlim = ax[1].get_xlim()
    ax1_ylim = ax[1].get_ylim()

    plt.close()  # Close static plot

    # Animation function
    def update(i):
        # Clear axes for each frame
        for a in ax:
            a.clear()

        if i >= len(distances):
            i = len(distances) - 1

        # Plot distance up to frame i
        ax[0].plot(
            np.arange(i + 1),
            distances[:i + 1],
            color=line_color,
            linewidth=3,
            label='Distance'
        )
        # Show the current frame in the second subplot
        ax[1].imshow(frames_tensor[i].permute(1, 2, 0) / 255)

        ax[0].set_xlim(ax0_xlim)
        ax[0].set_ylim(0, 50)  # Adjust if your distances differ in scale
        ax[1].set_xlim(ax1_xlim)
        ax[1].set_ylim(ax1_ylim)

        ax[0].set_title(f"{embedding_name} Embedding Distance", fontsize=15)
        ax[0].set_xlabel("Frame", fontsize=15)
        ax[0].set_ylabel("Distance", fontsize=15)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title("Current Video Frame", fontsize=15)
        ax[0].legend(loc="upper right")

        return ax

    # Create the FuncAnimation and save
    ani = FuncAnimation(fig, update, frames=len(distances) + 30, interval=20, repeat=False)
    ani.save(outpath, dpi=100, writer=PillowWriter(fps=25))

def main(rep='vip'):
    # Load embedding model + transform
    model, transform = load_embedding(rep)
    model.to('cuda')
    snapshot_path = "/home/emlyn/rl_franka/vip/vip/vipoutput/train_vip_finetune/2025-03-21_12-17-39/snapshot_8500.pt"
    load_snapshot(model, snapshot_path)
    model.eval()  # ensure eval mode

    embedding_names = {'vip': 'VIP', 'resnet': 'ResNet', 'r3m': 'R3M'}
    colors = {'vip': 'tab:blue', 'resnet': 'tab:orange', 'r3m':'tab:red'}

    emb_name = embedding_names.get(rep, 'VIP')
    color = colors.get(rep, 'tab:blue')

    os.makedirs('embedding_curves', exist_ok=True)

    # --- 1) Process success.mp4 for demonstration ---
    success_path = "strawb/success_2.mp4"
    success_frames = get_video_frames(success_path)
    success_embeddings, success_frames_tensor = get_embeddings(
        success_frames, model, transform, rep
    )

    # Instead of using the final frame's embedding as the goal:
    # Load your pre-saved embedding from mean_embedding.pkl
    import pickle
    with open("mean_embedding.pkl", "rb") as f:
        goal_embedding = pickle.load(f)

    # We'll still use the success video’s final frame just for *visual display* of a "goal frame"
    goal_frame_tensor = success_frames_tensor[-1]

    # --- 2) Process fail.mp4 using that loaded embedding as the goal ---
    fail_path = "strawb/success_2.mp4"
    fail_frames = get_video_frames(fail_path)
    fail_embeddings, fail_frames_tensor = get_embeddings(
        fail_frames, model, transform, rep
    )

    # Distances for the fail video using the loaded "goal_embedding"
    distances_fail = np.linalg.norm(fail_embeddings - goal_embedding, axis=1)

    # Animate and save fail results
    outpath_fail = f"embedding_curves/fail_vs_loaded_goal_{rep}.gif"
    animate_distance(
        distances=distances_fail,
        frames_tensor=fail_frames_tensor,
        goal_frame_tensor=goal_frame_tensor,  # Just for the "Goal Frame" display
        outpath=outpath_fail,
        rep=rep,
        embedding_name=emb_name,
        line_color=color
    )

if __name__ == '__main__':
    main(rep='vip')
