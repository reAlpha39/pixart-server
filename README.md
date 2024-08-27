## How to Install

#### 1.Setup your platform
-   miniconda (for python environment)
-   python (3.10 recommended)
-   pip
-   git
-   git lfs

#### 2. Setup environment
```
conda create -n pixart python=3.10
conda activate pixart
```

#### 3. Download Models
```
huggingface-cli download dataautogpt3/PixArt-Sigma-900M --local-dir ./PixArt-Sigma-900M
```

#### 4. Install dependency
```
pip install -r requirements.txt
```

#### 5. Run
```
fastapi run app.py
```
