import torch
from dataset import CycleGANDataset
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from utils.helpers import save_checkpoint, load_checkpoint, seed_everything
from models.definitions import Generator, Discriminator
from tqdm import tqdm
from torchvision.utils import save_image
import os

def train_fn(disc_A, disc_B, gen_AB, gen_BA, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    loop = tqdm(loader, leave=True)

    for idx, (A, B) in enumerate(loop):
        A = A.to(config.DEVICE)
        B = B.to(config.DEVICE)

        # Train Discriminators A and B
        with torch.cuda.amp.autocast(enabled=config.DEVICE=="cuda"):
            fake_A = gen_BA(B)
            D_A_real = disc_A(A)
            D_A_fake = disc_A(fake_A.detach())
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss

            fake_B = gen_AB(A)
            D_B_real = disc_B(B)
            D_B_fake = disc_B(fake_B.detach())
            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss

            D_loss = (D_A_loss + D_B_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators
        with torch.cuda.amp.autocast(enabled=config.DEVICE=="cuda"):
            # Adversarial loss for both generators
            D_A_fake = disc_A(fake_A)
            D_B_fake = disc_B(fake_B)
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))

            # Cycle loss
            cycle_A = gen_BA(fake_B)
            cycle_B = gen_AB(fake_A)
            cycle_A_loss = l1(A, cycle_A)
            cycle_B_loss = l1(B, cycle_B)

            # Identity loss (optional but good for preservation)
            id_A = gen_BA(A)
            id_B = gen_AB(B)
            id_A_loss = l1(A, id_A)
            id_B_loss = l1(B, id_B)

            G_loss = (
                loss_G_A
                + loss_G_B
                + cycle_A_loss * config.LAMBDA_CYCLE
                + cycle_B_loss * config.LAMBDA_CYCLE
                + id_A_loss * config.LAMBDA_IDENTITY
                + id_B_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            if not os.path.exists("saved_images"):
                os.makedirs("saved_images")
            save_image(fake_A * 0.5 + 0.5, f"saved_images/fake_A_{idx}.png")
            save_image(fake_B * 0.5 + 0.5, f"saved_images/fake_B_{idx}.png")

        loop.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item())

def main():
    disc_A = Discriminator(in_channels=3).to(config.DEVICE)
    disc_B = Discriminator(in_channels=3).to(config.DEVICE)
    gen_AB = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_BA = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_AB.parameters()) + list(gen_BA.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_AB, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_BA, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_A, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_B, opt_disc, config.LEARNING_RATE,
        )

    dataset = CycleGANDataset(
        root_image="data/train", # Using unified root for consistency with dataset.py
        transforms=config.TRANSFORMS,
    )
    val_dataset = CycleGANDataset(
        root_image="data/val",
        transforms=config.TRANSFORMS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler(enabled=config.DEVICE=="cuda")
    d_scaler = torch.cuda.amp.GradScaler(enabled=config.DEVICE=="cuda")

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_A,
            disc_B,
            gen_AB,
            gen_BA,
            loader,
            opt_disc,
            opt_gen,
            l1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_AB, opt_gen, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(gen_BA, opt_gen, filename=config.CHECKPOINT_GEN_B)
            save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_CRITIC_A)
            save_checkpoint(disc_B, opt_disc, filename=config.CHECKPOINT_CRITIC_B)

if __name__ == "__main__":
    main()
