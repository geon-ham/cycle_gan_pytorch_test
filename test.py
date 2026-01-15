import torch
from dataset import CycleGANDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import config
from utils.helpers import load_checkpoint
from models.definitions import Generator
from torchvision.utils import save_image
import os

def test():
    # Define Generator A -> B (Endoscope -> WSI)
    gen_AB = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    
    # We need an optimizer just to satisfy load_checkpoint signature, though we won't step it
    opt_gen = optim.Adam(list(gen_AB.parameters()), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    try:
        load_checkpoint(
            config.CHECKPOINT_GEN_A, gen_AB, opt_gen, config.LEARNING_RATE,
        )
    except FileNotFoundError:
        print(f"Checkpoint file {config.CHECKPOINT_GEN_A} not found. Ensure you have trained the model.")

    val_dataset = CycleGANDataset(
        root_image="data/val",
        transforms=config.TRANSFORMS,
    )
    
    loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    if not os.path.exists("saved_images/inference"):
        os.makedirs("saved_images/inference")

    gen_AB.eval()
    with torch.no_grad():
        for idx, (A, B) in enumerate(loader):
            A = A.to(config.DEVICE)
            fake_B = gen_AB(A)
            save_image(fake_B * 0.5 + 0.5, f"saved_images/inference/gen_{idx}.png")
            save_image(A * 0.5 + 0.5, f"saved_images/inference/input_{idx}.png")
            print(f"Saved inference image {idx}")

if __name__ == "__main__":
    test()
