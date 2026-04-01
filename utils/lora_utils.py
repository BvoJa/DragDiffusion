from PIL import Image
import os
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
import traceback # Thêm để in lỗi chi tiết

from transformers import AutoTokenizer, PretrainedConfig
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor, 
    AttnAddedKVProcessor2_0, 
    SlicedAttnAddedKVProcessor,
)
from diffusers.models.lora import LoRALinearLayer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import unet_lora_state_dict

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision,
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def tokenize_prompt(tokenizer, prompt):
    return tokenizer(prompt, truncation=True, padding="max_length", 
                     max_length=tokenizer.model_max_length, return_tensors="pt")

def encode_prompt(text_encoder, input_ids, attention_mask):
    prompt_embeds = text_encoder(input_ids.to(text_encoder.device), 
                                 attention_mask=attention_mask.to(text_encoder.device) if attention_mask is not None else None)
    return prompt_embeds[0]

def train_lora(image, prompt, model_path, vae_path, save_lora_path, lora_step, lora_lr, lora_batch_size, lora_rank, progress, save_interval=-1):
    try:
        print(f"🚀 Khởi tạo training: Steps={lora_step}, LR={lora_lr}, Rank={lora_rank}")
        accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision='fp16')
        set_seed(0)

        # Load models
        print("⏬ Đang nạp Tokenizer & Scheduler...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", use_fast=False)
        noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        
        print("⏬ Đ đang nạp Text Encoder & UNet...")
        text_encoder_cls = import_model_class_from_model_name_or_path(model_path, revision=None)
        text_encoder = text_encoder_cls.from_pretrained(model_path, subfolder="text_encoder")
        
        if vae_path == "default":
            vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
        else:
            vae = AutoencoderKL.from_pretrained(vae_path)
        
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")

        # Đóng băng tham số gốc
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)

        device = accelerator.device
        unet.to(device, dtype=torch.float16)
        vae.to(device, dtype=torch.float16)
        text_encoder.to(device, dtype=torch.float16)

        # --- TIÊM LORA (Injecting LoRA) ---
        print("💉 Đang tiêm lớp LoRA vào UNet...")
        unet_lora_parameters = []
        for attn_processor_name, attn_processor in unet.attn_processors.items():
            attn_module = unet
            for n in attn_processor_name.split(".")[:-1]:
                attn_module = getattr(attn_module, n)

            # ĐÃ SỬA: Thay args.rank bằng lora_rank
            attn_module.to_q.set_lora_layer(LoRALinearLayer(in_features=attn_module.to_q.in_features, out_features=attn_module.to_q.out_features, rank=lora_rank))
            attn_module.to_k.set_lora_layer(LoRALinearLayer(in_features=attn_module.to_k.in_features, out_features=attn_module.to_k.out_features, rank=lora_rank))
            attn_module.to_v.set_lora_layer(LoRALinearLayer(in_features=attn_module.to_v.in_features, out_features=attn_module.to_v.out_features, rank=lora_rank))
            attn_module.to_out[0].set_lora_layer(LoRALinearLayer(in_features=attn_module.to_out[0].in_features, out_features=attn_module.to_out[0].out_features, rank=lora_rank))

            unet_lora_parameters.extend(attn_module.to_q.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.to_k.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.to_v.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.to_out[0].lora_layer.parameters())

        # Optimizer
        optimizer = torch.optim.AdamW(unet_lora_parameters, lr=lora_lr)
        lr_scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=lora_step)

        unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)

        # Encode prompt
        text_inputs = tokenize_prompt(tokenizer, prompt)
        text_embedding = encode_prompt(text_encoder, text_inputs.input_ids, text_inputs.attention_mask)
        text_embedding = text_embedding.repeat(lora_batch_size, 1, 1).to(device, dtype=torch.float16)

        # Image transforms
        image_transforms = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        print("🔥 Bắt đầu vòng lặp training...")
        for step in progress.tqdm(range(lora_step), desc="Training LoRA"):
            unet.train()
            
            # Chuẩn bị batch ảnh
            img_pil = Image.fromarray(image)
            image_tensor = image_transforms(img_pil).unsqueeze(0).repeat(lora_batch_size, 1, 1, 1).to(device, dtype=torch.float16)

            # Encode ảnh sang latent
            latents = vae.encode(image_tensor).latent_dist.sample() * vae.config.scaling_factor
            
            # Khuyếch tán (Diffusion process)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (lora_batch_size,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Dự đoán nhiễu
            model_pred = unet(noisy_latents, timesteps, text_embedding).sample
            
            # Tính Loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % 20 == 0:
                print(f"Step {step}/{lora_step} | Loss: {loss.item():.4f}")

        # --- LƯU KẾT QUẢ ---
        print(f"💾 Đang lưu LoRA vào: {save_lora_path}")
        if not os.path.exists(save_lora_path):
            os.makedirs(save_lora_path)
            
        unet_lora_layers = unet_lora_state_dict(unet)
        LoraLoaderMixin.save_lora_weights(save_directory=save_lora_path, unet_lora_layers=unet_lora_layers, text_encoder_lora_layers=None)
        print("✅ TRAINING HOÀN TẤT!")

    except Exception as e:
        print("❌ LỖI TRONG train_lora:")
        traceback.print_exc()
        raise e