# training/losses.py

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import LAMBDA_GP, DEVICE


def discriminator_loss(D, real_images, fake_images, attrs):
    """
    WGAN-GP Discriminator (Critic) loss.

    The Critic wants to:
        - assign high scores to real images
        - assign low scores to fake images
        - satisfy the Lipschitz constraint (enforced via gradient penalty)

    Loss = mean(fake_scores) - mean(real_scores) + lambda * gradient_penalty

    Args:
        D           : Discriminator model
        real_images : [batch, 3, 64, 64] — real samples from dataset
        fake_images : [batch, 3, 64, 64] — generated samples from Generator
        attrs       : [batch, NUM_ATTRIBUTES] — attribute conditioning vector

    Returns:
        loss  (scalar),
        real_score (scalar, for logging),
        fake_score (scalar, for logging)
    """
    real_scores = D(real_images, attrs)           # [batch, 1]
    fake_scores = D(fake_images.detach(), attrs)  # detach: don't backprop into Generator here

    gp   = gradient_penalty(D, real_images, fake_images, attrs)
    loss = fake_scores.mean() - real_scores.mean() + LAMBDA_GP * gp

    return loss, real_scores.mean().item(), fake_scores.mean().item()


def generator_loss(D, fake_images, attrs):
    """
    WGAN-GP Generator loss.

    The Generator wants the Critic to assign high scores to its fake images.
    So it minimizes the negative of the fake scores.

    Loss = -mean(fake_scores)

    Args:
        D           : Discriminator model
        fake_images : [batch, 3, 64, 64] — generated samples
        attrs       : [batch, NUM_ATTRIBUTES]

    Returns:
        loss (scalar)
    """
    fake_scores = D(fake_images, attrs)   # no detach — we need gradients through Generator
    loss = -fake_scores.mean()
    return loss


def gradient_penalty(D, real_images, fake_images, attrs):
    """
    WGAN-GP Gradient Penalty.

    Enforces the 1-Lipschitz constraint on the Critic by penalizing
    gradients whose norm deviates from 1, computed on interpolated samples.

    Steps:
        1. Create interpolated images: x_hat = alpha * real + (1 - alpha) * fake
        2. Compute Critic scores on x_hat
        3. Compute gradients of scores w.r.t. x_hat
        4. Penalize (||gradients||_2 - 1)^2

    Args:
        D           : Discriminator model
        real_images : [batch, 3, 64, 64]
        fake_images : [batch, 3, 64, 64]
        attrs       : [batch, NUM_ATTRIBUTES]

    Returns:
        penalty (scalar)
    """
    batch_size = real_images.size(0)

    # Random interpolation weight, one per sample in the batch
    alpha = torch.rand(batch_size, 1, 1, 1, device=DEVICE)   # [batch, 1, 1, 1]

    # Interpolate between real and fake
    interpolated = (alpha * real_images + (1 - alpha) * fake_images.detach()).requires_grad_(True)

    # Critic score on interpolated images
    interp_scores = D(interpolated, attrs)   # [batch, 1]

    # Compute gradients w.r.t. interpolated images
    gradients = torch.autograd.grad(
        outputs=interp_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interp_scores),
        create_graph=True,      # needed to backprop through the penalty itself
        retain_graph=True,
        only_inputs=True
    )[0]   # [batch, 3, 64, 64]

    # Flatten gradients per sample, compute L2 norm
    gradients = gradients.view(batch_size, -1)                # [batch, 3*64*64]
    gradient_norm = gradients.norm(2, dim=1)                  # [batch]

    # Penalty: (||grad|| - 1)^2, averaged over the batch
    penalty = ((gradient_norm - 1) ** 2).mean()

    return penalty


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import LATENT_DIM, NUM_ATTRIBUTES, CHANNELS, IMAGE_SIZE
    from models.generator import Generator
    from models.discriminator import Discriminator

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    real   = torch.randn(4, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    attrs  = torch.randint(0, 2, (4, NUM_ATTRIBUTES)).float().to(DEVICE)
    z      = torch.randn(4, LATENT_DIM).to(DEVICE)
    fake   = G(z, attrs)

    d_loss, real_score, fake_score = discriminator_loss(D, real, fake, attrs)
    g_loss = generator_loss(D, fake, attrs)

    print(f"D loss       : {d_loss.item():.4f}")
    print(f"Real score   : {real_score:.4f}")
    print(f"Fake score   : {fake_score:.4f}")
    print(f"G loss       : {g_loss.item():.4f}")
