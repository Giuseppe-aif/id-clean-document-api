from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "ID Clean Document API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
