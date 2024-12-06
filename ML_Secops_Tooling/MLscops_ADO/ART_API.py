from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import tensorflow as tf
from art.attacks.evasion import FastGradientMethod
from art.attacks.inference import MembershipInferenceBlackBox
from art.attacks.extraction import CopycatCNN
from art.attacks.poisoning import PoisoningAttackSVM
from art.estimators.classification import SklearnClassifier, TensorFlowV2Classifier
from art.utils import load_dataset
from PIL import Image
import io

# FastAPI instance
app = FastAPI()

# Global variables for models and ART estimators
image_classifier = None
xgb_classifier = None
tensorflow_model = None

@app.on_event("startup")
def load_models():
    """
    Load models and wrap them with ART estimators.
    """
    global image_classifier, xgb_classifier, tensorflow_model

    # Load XGBoost model
    xgb_model = XGBClassifier()
    # Assume you have a pre-trained model file: "xgb_model.json"
    xgb_model.load_model("xgb_model.json")
    xgb_classifier = SklearnClassifier(model=xgb_model)

    # Load TensorFlow model for image classification
    tensorflow_model = tf.keras.models.load_model("image_model.h5")
    image_classifier = TensorFlowV2Classifier(model=tensorflow_model, nb_classes=10, input_shape=(28, 28, 1))

# Pydantic schema for input data
class EvasionInput(BaseModel):
    image: list
    eps: float = 0.1

class PoisoningInput(BaseModel):
    data: list
    labels: list

@app.post("/evasion/image/")
def evasion_attack(data: EvasionInput):
    """
    Perform evasion attack on an image using FGSM.
    """
    try:
        global image_classifier
        if image_classifier is None:
            raise HTTPException(status_code=500, detail="Image classifier not loaded")

        images = np.array(data.image).astype(np.float32)
        fgsm = FastGradientMethod(estimator=image_classifier, eps=data.eps)
        adversarial_images = fgsm.generate(x=images)

        return {"adversarial_images": adversarial_images.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evasion/xgboost/")
def evasion_attack_xgb(data: PoisoningInput):
    """
    Perform evasion attack on XGBoost using Poisoning Attack.
    """
    try:
        global xgb_classifier
        if xgb_classifier is None:
            raise HTTPException(status_code=500, detail="XGBoost classifier not loaded")

        attack = PoisoningAttackSVM(classifier=xgb_classifier)
        adv_data, adv_labels = attack.generate(x=np.array(data.data), y=np.array(data.labels))

        return {"adversarial_data": adv_data.tolist(), "adversarial_labels": adv_labels.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference/")
def inference_attack(data: PoisoningInput):
    """
    Perform membership inference attack.
    """
    try:
        global image_classifier
        if image_classifier is None:
            raise HTTPException(status_code=500, detail="Image classifier not loaded")

        attack = MembershipInferenceBlackBox(image_classifier)
        result = attack.infer(x=np.array(data.data), y=np.array(data.labels))

        return {"membership_inference_result": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extraction/")
def extraction_attack(model: str, file: UploadFile = File(...)):
    """
    Perform model extraction attack using CopycatCNN.
    """
    try:
        if model not in ["image_classifier", "xgboost"]:
            raise HTTPException(status_code=400, detail="Invalid model type specified")

        file_content = io.BytesIO(file.file.read())
        images = np.array(Image.open(file_content)).astype(np.float32)

        if model == "image_classifier" and image_classifier is not None:
            attack = CopycatCNN(image_classifier, nb_epochs=5, nb_stolen=10)
            stolen_model = attack.extract(x=images)
            return {"message": "Model extraction complete", "stolen_model_summary": str(stolen_model.summary())}
        else:
            raise HTTPException(status_code=500, detail="Model not available for extraction")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))