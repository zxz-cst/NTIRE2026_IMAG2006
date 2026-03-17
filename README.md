# NTIRE 2026 Challenge on Mobile Real-World Image Super-Resolution @ CVPR 2026

**Team:** IMAG2006

This repository provides the inference code for our NTIRE 2026 Mobile Real-World Image Super-Resolution solution.

## 1. Download the required weights

Before running the code, please download the following files and prepare their local paths.

### 1.1 Stable Diffusion 2.1 Base
-  [sd2_1_base](https://pan.baidu.com/s/1mCnUjlfwjvkYIGiSi1xo2w 提取码: rx27)

### 1.2 LoRA weights
- [LoRA_weights](https://pan.baidu.com/s/1WjB2v2pk2HN3nQgN1LgIUw 提取码: ewp6)

### 1.3 Empty prompt embeddings
- [empty_prompt_embeds.pt]( https://pan.baidu.com/s/1Z2FX5h8x5-wxi9EHufWMmQ 提取码: sbeh)

​	Our method does not require prompts; it only needs to load an empty prompt to satisfy the input requirement of SD 2.1 base and can then work properly.

---

## 2. Set up the environment

We recommend creating a clean conda environment first.

```bash
conda create -n IMAG2006 python=3.10 -y
conda activate IMAG2006
pip install -r requirements.txt
```

---

## 3. Modify the config file

Edit the following file:

```bash
NTIRE2026_Mobile_RealWorld_ImageSR/model_zoo/team06_IMAG2006/config.json
```

Use the correct local paths for the downloaded weights:

```json
{
  "sd_path": "/path/to/sd2.1_base",
  "lora_path": "/path/to/lora_model",
  "prompt": "",
  "prompt_embeds_path": "/path/to/empty_prompt_embeds.pt",
  "mid_timestep": 273,
  "process_size": 512,
  "upscale": 4,
  "align_method": "wavelet",
  "weight_dtype": "bf16",
  "png_compress_level": 0
}
```

### Parameter notes
- `sd_path`: local path to the Stable Diffusion 2.1 Base model
- `lora_path`: local path to the LoRA weights
- `prompt_embeds_path`: local path to `empty_prompt_embeds.pt`
- `mid_timestep`: one-step diffusion timestep used during inference
- `process_size`: inference patch size
- `upscale`: upscaling factor
- `align_method`: latent/image alignment method
- `weight_dtype`: inference precision, e.g. `bf16`
- `png_compress_level`: PNG compression level for saved results

---

## 4. Run the inference code

Use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --valid_dir /path/to/valid_dir \
    --test_dir /path/to/test_dir \
    --save_dir /path/to/save_dir \
    --model_id 6
```

### Usage notes
- You may use `--valid_dir` only, `--test_dir` only, or both.
- Please make sure to replace `/path/to/valid_dir`, `/path/to/test_dir`, and `/path/to/save_dir` with your actual paths.
- The output images will be saved to the directory specified by `--save_dir`.

---

## 5. Quick start

If you only want the minimal workflow, follow these four steps:

1. Download `sd2_1_base`, `lora_model`, and `empty_prompt_embeds.pt`.
2. Create the conda environment and install dependencies.
3. Update `config.json` with your local model paths.
4. Run `test.py` with the desired input and output directories.

---

