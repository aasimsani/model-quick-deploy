from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
import io
# from starlette.responses import StreamingResponse

import cv2
import numpy as np

import torch
import cv2
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Load model

# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Code from: https://fastapi.tiangolo.com/tutorial/request-files/
app = FastAPI()

@app.post("/uploadfiles/")

async def create_upload_files(files: List[UploadFile] = File(...)):

	for image in files:
		img = cv2.imdecode(np.frombuffer(image.file.read(), np.uint8), cv2.IMREAD_COLOR)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		input_batch = transform(img).to(device)

		with torch.no_grad():
			prediction = midas(input_batch)

			prediction = torch.nn.functional.interpolate(
				prediction.unsqueeze(1),
				size=img.shape[:2],
				mode="bicubic",
				align_corners=False,
			).squeeze()

		fig = Figure()
		canvas = FigureCanvas(fig)
		ax = fig.gca()

		output = prediction.cpu().numpy()
		ax.imshow(img)
		ax.imshow(output, cmap="jet", alpha=0.9)
		ax.axis("off")
		canvas.draw()
		canvas.show()
		output_image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
		res, im_png = cv2.imencode(".jpg", output_image)
		return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpg")

	# 	# # plt.imshow(img)	
	# 	# plt.imshow(output, alpha=0.8)
	# plt.show()
	return {"filenames": [file.filename for file in files]}


@app.get("/")
async def main():
	content = """
	<body>
		<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
			<input name="files" type="file" multiple>
			<input type="submit">
		</form>
	</body>
	"""
	return HTMLResponse(content=content)
