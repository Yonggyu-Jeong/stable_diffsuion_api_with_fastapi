import uvicorn
from fastapi import FastAPI
from app.image import image_router


app = FastAPI()

app.include_router(image_router.router_image, prefix="/image", tags=["image"])


@app.get("/")
async def root():
    return {"message": "Hello, thank you for visiting my 'stable diffusion api with fastapi project. "}


# uvicorn 사용하여 FastAPI 실행
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
