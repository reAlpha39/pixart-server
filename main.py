import uvicorn
from modules.api import app as Api

if __name__ == "__main__":
    uvicorn.run(Api, host=["::", "0.0.0.0"], port=8000)
