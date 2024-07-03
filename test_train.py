
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
import glob
import pdb
import wandb

from data import Data
from PIL import Image
from dataclasses import dataclass
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from accelerate import Accelerator, notebook_launcher
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
from accelerate import notebook_launcher
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.utils.convert_parameters as cp

# start a new wandb run to track this script



@dataclass
class TrainingConfig:
    image_size_1 = 256  # the generated image resolution
    image_size_2 = 128  # the generated image resolution
    train_batch_size = 32
    eval_batch_size = 32  # how many images to sample during evaluation
    num_epochs = 2000
    gradient_accumulation_steps = 1
    learning_rate = 5e-5
    lr_warmup_steps = 500
    save_image_epochs = 100
    save_model_epochs = 2000
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-Market-384-128-2000epoch-wsd"  # the model name locally and on the HF Hub
    
    wandb_project_name = "CC-ReID"
    wandb_run_name = "ddpm-Market1501"

    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_model_id = "DDOING2/ddpm"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 48


config = TrainingConfig()


""" 
config.dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(config.dataset_name, split="train")
 """

dataset = Data()
""" fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for i, image in enumerate(dataset[:4]["image"]):
    axs[i].imshow(image)
    axs[i].set_axis_off()
fig.show() """


model = UNet2DModel(
    sample_size=(config.image_size_1,config.image_size_2) , # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(64, 64, 128, 128, 256, 256),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)
""" 
sample_image = dataset[0]["images"].unsqueeze(0)
print("Input shape:", sample_image.shape)

print("Output shape:", model(sample_image, timestep=0).sample.shape)

 """

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
""" noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

 """


""" noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)
 """


optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(dataset.train_loader) * config.num_epochs),
)



def evaluate(config, epoch, pipeline):
    images = []
    batch_size = config.eval_batch_size
    num_images_needed = 1
    num_batches = (num_images_needed + batch_size - 1) // batch_size

    for _ in range(num_batches):
        batch_images = pipeline(
            batch_size=batch_size,
            generator=torch.Generator(device='cpu').manual_seed(config.seed),
        ).images
        images.extend(batch_images)

    images = images[:num_images_needed]

    if len(images) != num_images_needed:
        raise ValueError(f"Expected {num_images_needed} images, but got {len(images)}")

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)

    for i, image in enumerate(images):
        image_path = os.path.join(test_dir, f"{epoch:04d}_{i:02d}.png")
        image.save(image_path)
    
    
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    entity = "ggara376",
    project="CC-ReID",
    name = "ddpm-Market1501",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 5e-5,
    "architecture": "DDPM",
    "dataset": "Market-1501",
    "epochs": 2000,
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "input_size": (384,128),
    "lr_warmup_steps": 500,
    "mixed_precision": "fp16",
    "seed": 48
    }
)  
wandb.init(
            entity = "ggara376",
            project=config.wandb_project_name,  # WandB 프로젝트 이름
            name=config.wandb_run_name,  # WandB 실행 이름
            dir=config.output_dir,  # 로그 저장 디렉토리
            config=config  # 설정 정보를 WandB에 로깅
        )

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("CC-ReID")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, dataset.train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataset.train_loader, lr_scheduler
    )
    #model = DDP(model, device_ids=[accelerator.local_process_index])
    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(dataset.train_loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        epoch_loss = 0
        log_dict = {}
        for step, batch in enumerate(dataset.train_loader):
            images, labels = batch
            clean_images = images
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                epoch_loss += loss
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)
                    
            # Save model checkpoint
            if epoch == config.num_epochs - 1:
                model_path = os.path.join(config.output_dir, "final_model.pth")
                torch.save(model.state_dict(), model_path)
                    
    
                    


args = (config, model, noise_scheduler, optimizer, dataset.train_loader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)



sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1])