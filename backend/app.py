from flask import Flask, jsonify, request
from model import Net
from torchvision import transforms
import torch
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

model = Net()
pre_trained_model = torch.load('./mnist_cnn.pt')
model.load_state_dict(pre_trained_model, strict=False)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    if data is None:
        print("No valid request body, json missing!")
        return jsonify({'error': 'No valid request body, json missing!'})
    else:
        img_data = data['base64']

        img = convert_from_base64(img_data)

        _, prediction = model.forward(transform(img).unsqueeze(0)).max(1)
        return {'prediction': prediction.item(), 'class_name': str(prediction.item())}

def convert_from_base64(b64_string):

    starter = b64_string.find(',')
    image_data = b64_string[starter+1:]
    image_data = bytes(image_data, encoding="ascii")
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    return im

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')