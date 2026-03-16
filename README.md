## [NTIRE 2026 Challenge on Mobile Real-World Image Super-Resolution](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/) TEAM IMAG2006

### DownLoad

You need download:

sd2.1base

 [sd-2-1-base](https://pan.baidu.com/s/1mCnUjlfwjvkYIGiSi1xo2w 提取码: rx27)

lora_model

empty_emd

[empty_prompt_embeds.pt]( https://pan.baidu.com/s/1Z2FX5h8x5-wxi9EHufWMmQ 提取码: sbeh)



### Environment requirements

```
conda create -n IMAG2006 python=3.10
conda activate IMAG2006
pip install -r requirements.txt
```

### Config

```
NTIRE2026_Mobile_RealWorld_ImageSR/model_zoo/team06_IMAG2006/config.json
{
  "sd_path": "sd2.1_base path",
  "lora_path": "lora weight path",
  "prompt": "",
  "prompt_embeds_path": "empty_prompt_embeds.pt path",
  "mid_timestep": 273,
  "process_size": 512,
  "upscale": 4,
  "align_method": "wavelet",
  "weight_dtype": "bf16",
  "png_compress_level": 0
}
```



### Running commands

```
CUDA_VISIBLE_DEVICES=0 python test.py
    --valid_dir [path to val data dir] \
    --test_dir [path to test data dir] \
    --save_dir [path to your save dir] \
    --model_id 6
```

- You can use either `--valid_dir`, or `--test_dir`, or both of them. Be sure the change the directories `--valid_dir`/`--test_dir` and `--save_dir`.

