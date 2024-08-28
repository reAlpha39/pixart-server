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
```

Then put those folder on the "**root**" directory

#### 6. Run
```
fastapi run app.py
```
