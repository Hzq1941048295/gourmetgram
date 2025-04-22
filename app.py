
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
from mimetypes import guess_type # used to identify the type of image
from datetime import datetime # used to generate timestamp tag for image
import uuid # used to generate unique ID per image
import boto3 # client for s3-compatible object store, including MinIO
from concurrent.futures import ThreadPoolExecutor  # used for the thread pool that will upload images to MinIO
executor = ThreadPoolExecutor(max_workers=2)  # can adjust max_workers as needed

# New! Authenticate to MinIO object store
s3 = boto3.client(
    's3',
    endpoint_url=os.environ['MINIO_URL'],  # e.g. 'http://minio:9000'
    aws_access_key_id=os.environ['MINIO_USER'],
    aws_secret_access_key=os.environ['MINIO_PASSWORD'],
    region_name='us-east-1'  # required for the boto client but not used by MinIO
)

app = Flask(__name__)

os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)

model = torch.load("food11.pth", map_location=torch.device('cpu') )

# New! for uploading production images to MinIO bucket
def upload_production_bucket(img_path, preds, confidence, prediction_id):
    classes = np.array(["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
	    "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
	    "Vegetable/Fruit"])
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    pred_index = np.where(classes == preds)[0][0]
    class_dir = f"class_{pred_index:02d}"

    bucket_name = "production"
    root, ext = os.path.splitext(img_path)
    content_type = guess_type(img_path)[0] or 'application/octet-stream'
    s3_key = f"{class_dir}/{prediction_id}{ext}"
    
    with open(img_path, 'rb') as f:
        s3.upload_fileobj(f, 
            bucket_name, 
            s3_key, 
            ExtraArgs={'ContentType': content_type}
            )

    # tag the object with predicted class and confidence
    s3.put_object_tagging(
        Bucket=bucket_name,
        Key=s3_key,
        Tagging={
            'TagSet': [
                {'Key': 'predicted_class', 'Value': preds},
                {'Key': 'confidence', 'Value': f"{confidence:.3f}"},
                {'Key': 'timestamp', 'Value': timestamp}
            ]
        }
    )
	
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def model_predict(img_path, model):
    img = Image.open(img_path).convert('RGB')
    img = preprocess_image(img)

    classes = np.array(["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
	"Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
	"Vegetable/Fruit"])

    with torch.no_grad():
        output = model(img)
        prob, predicted_class = torch.max(output, 1)
    
    return classes[predicted_class.item()], torch.sigmoid(prob).item()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    preds = None
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        f.save(os.path.join(app.instance_path, 'uploads', secure_filename(f.filename)))
        preds, probs = model_predict("./instance/uploads/" + secure_filename(f.filename), model)
        return '<button type="button" class="btn btn-info btn-sm">' + str(preds) + '</button>' 
    return '<a href="#" class="badge badge-warning">Warning</a>'

@app.route('/test', methods=['GET'])
def test():
    preds, probs = model_predict("./instance/uploads/test_image.jpeg", model)
    return str(preds)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
