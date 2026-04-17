# training/train.py

import os
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import (
    DEVICE, LATENT_DIM, NUM_EPOCHS, BATCH_SIZE,
    LR_G, LR_D, BETA1, BETA2,
    CRITIC_STEPS, SAMPLE_INTERVAL,
    CHECKPOINT_DIR, LOG_DIR
)
from data.dataset import get_celeba_dataloader
from models.generator import Generator
from models.discriminator import Discriminator
from training.losses import discriminator_loss, generator_loss
from utils.visualize import save_image_grid, plot_losses


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(G, D, opt_G, opt_D, epoch, step):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"ckpt_epoch{epoch:03d}_step{step:07d}.pt")
    torch.save({
        "epoch"       : epoch,
        "step"        : step,
        "G_state"     : G.state_dict(),
        "D_state"     : D.state_dict(),
        "opt_G_state" : opt_G.state_dict(),
        "opt_D_state" : opt_D.state_dict(),
    }, path)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(path, G, D, opt_G, opt_D):
    ckpt = torch.load(path, map_location=DEVICE)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    opt_G.load_state_dict(ckpt["opt_G_state"])
    opt_D.load_state_dict(ckpt["opt_D_state"])
    print(f"  Resumed from checkpoint: epoch {ckpt['epoch']}, step {ckpt['step']}")
    return ckpt["epoch"], ckpt["step"]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(resume_from=None):
    """
    Full WGAN-GP training loop for the Conditional DCGAN on CelebA.

    Args:
        resume_from (str): path to a checkpoint .pt file to resume training.
                           Pass None to start fresh.
    """

    # --- Data ---
    print("Loading CelebA dataset...")
    dataloader, attr_names = get_celeba_dataloader(split="train")
    print(f"Dataset ready. {len(dataloader)} batches per epoch.\n")

    # --- Models ---
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    # --- Optimizers ---
    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    # --- Resume from checkpoint if provided ---
    start_epoch = 0
    global_step = 0
    if resume_from:
        start_epoch, global_step = load_checkpoint(resume_from, G, D, opt_G, opt_D)

    # --- Loss tracking ---
    g_losses = []
    d_losses = []

    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"Starting training on {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}  |  Batch size: {BATCH_SIZE}  |  Critic steps: {CRITIC_STEPS}\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        for batch_idx, (real_images, attrs) in enumerate(dataloader):

            real_images = real_images.to(DEVICE)          # [batch, 3, 64, 64]
            attrs       = attrs.float().to(DEVICE)        # [batch, 40]

            # ----------------------------------------------------------------
            # Phase 1: Train Discriminator (Critic) for CRITIC_STEPS steps
            # ----------------------------------------------------------------
            for _ in range(CRITIC_STEPS):
                z          = torch.randn(BATCH_SIZE, LATENT_DIM, device=DEVICE)
                fake_images = G(z, attrs)

                d_loss, real_score, fake_score = discriminator_loss(
                    D, real_images, fake_images, attrs
                )

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            # ----------------------------------------------------------------
            # Phase 2: Train Generator (once per CRITIC_STEPS Critic updates)
            # ----------------------------------------------------------------
            z           = torch.randn(BATCH_SIZE, LATENT_DIM, device=DEVICE)
            fake_images = G(z, attrs)

            g_loss = generator_loss(D, fake_images, attrs)

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            # ----------------------------------------------------------------
            # Logging
            # ----------------------------------------------------------------
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            global_step += 1

            if global_step % 100 == 0:
                print(
                    f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}] "
                    f"Step [{global_step:07d}] "
                    f"D loss: {d_loss.item():+.4f}  "
                    f"G loss: {g_loss.item():+.4f}  "
                    f"Real score: {real_score:+.4f}  "
                    f"Fake score: {fake_score:+.4f}"
                )

            # Save image grid every SAMPLE_INTERVAL steps
            if global_step % SAMPLE_INTERVAL == 0:
                save_image_grid(G, step=global_step)
                plot_losses(g_losses, d_losses)

        # Save checkpoint at end of every epoch
        save_checkpoint(G, D, opt_G, opt_D, epoch=epoch + 1, step=global_step)

    print("\nTraining complete.")
    save_image_grid(G, step=global_step)
    plot_losses(g_losses, d_losses)

    # Save final model weights separately for easy loading in the Streamlit app
    final_G_path = os.path.join(CHECKPOINT_DIR, "generator_final.pt")
    torch.save(G.state_dict(), final_G_path)
    print(f"Final Generator weights saved → {final_G_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Conditional DCGAN on CelebA")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume training from"
    )
    args = parser.parse_args()

    train(resume_from=args.resume)
