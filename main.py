import uvicorn
from modules.api import app as Api

if __name__ == "__main__":
    uvicorn.run(Api, host="::", port=8000)
