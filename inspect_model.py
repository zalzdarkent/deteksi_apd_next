# import the inference-sdk
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="59znzY448C1kIAKz9kyv"
)

result = CLIENT.infer(your_image.jpg, model_id="detek-apd/5")