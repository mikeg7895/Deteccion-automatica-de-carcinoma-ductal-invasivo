from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()
templates = Jinja2Templates(directory="templates")
model = torch.load("model_mobile.pth")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )

@app.post("/predict")
async def predict(imageHispa: UploadFile = File(...)):
    image = await imageHispa.read()
    pil_image = Image.open(io.BytesIO(image)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor()
    ])
    input_tensor = transform(pil_image).unsqueeze(0)
    input_tensor = input_tensor.to("cuda")
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)
        prediction = torch.argmax(output, 1).item()
    value = "IDC -" if prediction == 0 else "IDC +"
    return {"message": value}