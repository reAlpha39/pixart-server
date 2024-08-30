## How to Install

#### 1.Setup your platform
-   miniconda (for python environment)
-   git
-   git lfs
-   [Cuda Toolkit](https://developer.nvidia.com/cuda-12-4-0-download-archive)

#### 2. Clone repository
```
git clone https://github.com/reAlpha39/pixart-server.git
```

#### 3. Setup environment
```
conda create -n pixart python=3.10
conda activate pixart
```

#### 4. Install dependency
```
pip install -r requirements.txt
```

#### 5. Download Models
```
huggingface-cli download dataautogpt3/PixArt-Sigma-900M --local-dir ./PixArt-Sigma-900M
huggingface-cli download PixArt-alpha/PixArt-Sigma-XL-2-1024-MS --local-dir ./PixArt-Sigma-XL-2-1024-MS
huggingface-cli download PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers --local-dir ./pixart_sigma_sdxlvae_T5_diffusers
```

Then put those folder on the "**root**" directory

#### 6. Run
```
python main.py
```
