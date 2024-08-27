## How to Install

#### 1.Setup your platform
-   miniconda (for python environment)
-   git
-   git lfs

#### 2. Clone repository
```
git clone https://github.com/reAlpha39/pixart-server.git
```

#### 3. Setup environment
```
conda create -n pixart python=3.10
conda activate pixart
```

#### 4. Download Models
```
huggingface-cli download dataautogpt3/PixArt-Sigma-900M --local-dir ./PixArt-Sigma-900M
```

Then put those folder on the "**root**" directory

#### 5. Install dependency
```
pip install -r requirements.txt
```

#### 6. Run
```
fastapi run app.py
```
