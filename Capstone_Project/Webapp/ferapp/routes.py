from ferapp import app 
from flask import render_template, request
from scipy import misc
import numpy as np 
import torch 
from torchvision import transforms
import os



activation = torch.nn.Softmax(dim=1)
indx2emotion = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
model = torch.jit.load(os.path.join(app.root_path, 'model_data/model_fer_acc_jit.pth'))
model.eval()

def transform_image(image):
    my_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(), transforms.Normalize([0.5077], [0.255])])
    return my_transforms(image).unsqueeze(0)

def get_prediction(image):
    image = transform_image(image)
    try:
        with torch.no_grad():
            pred_face = activation(model(image))
        topp, topk = pred_face.topk(1, dim=1)
        topp = topp.item()
        topk = topk.item()
        topp = "%.2f" %(topp)
        return indx2emotion[int(topk)], topp
    except Exception:
        return 'N/A', '-'

@app.route("/predict", methods=['POST', 'GET'])
def make_prediction():
    if request.method=='POST':
        if 'image' in request.files:
            file = request.files['image']
        else:
            file = None
        if not file: 
            return render_template('predict.html', error="No image found.")
        else:
            img = np.expand_dims(misc.imread(file, mode='L'), axis=-1)
            emotion_class, emotion_prob = get_prediction(img)
            return render_template('predict.html', emotion_pred_class=emotion_class, emotion_pred_score=emotion_prob)
    else:
        return render_template('predict.html', error="N/A")
    
@app.route('/')
def predict():
    return render_template('predict.html')

    
