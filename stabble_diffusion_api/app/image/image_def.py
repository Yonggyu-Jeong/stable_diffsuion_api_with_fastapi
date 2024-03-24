import torch
import base64
import diffusers
import json
from PIL import Image
from io import BytesIO
from typing import Optional
from fastapi import HTTPException, Depends, UploadFile
from pydantic import BaseModel
from starlette.responses import JSONResponse


class ImgConfig:
    def __init__(self, hf_token, custom_models):
        self.hf_token = hf_token
        self.custom_models = custom_models


# TODO ImgConfig 표준화및 수정
'''
    def __init__(self, hf_token, custom_models, url_path: Optional[str] = None):
        self.hf_token = hf_token
        self.custom_models = custom_models
        self.url_path = url_path
'''


class ImageInfo:
    def __init__(self, image_base64, seed, mime_type, nsfw):
        self.base64 = image_base64
        self.seed = seed
        self.mime_type = mime_type
        self.nsfw = nsfw


class TaskParams(BaseModel):
    seed: int = 0
    num_outputs: int = 1
    prompt: Optional[str] = None
    init_image: Optional[UploadFile] = None
    mask_image: Optional[UploadFile] = None
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    eta: float = 0.0
    width: int = 512
    height: int = 512
    strength: float = 0.7

    def __init__(self, data_model: Optional[BaseModel] = None, **kwargs):
        super().__init__(**kwargs)
        if data_model is not None:
            self.process_model(data_model)

    def process_model(self, data_model: BaseModel):
        self.seed = data_model.seed
        self.num_outputs = data_model.num_outputs
        self.prompt = data_model.prompt
        self.init_image = data_model.init_image
        self.mask_image = data_model.mask_image
        self.num_inference_steps = data_model.num_inference_steps
        self.guidance_scale = data_model.guidance_scale
        self.eta = data_model.eta
        self.width = data_model.width
        self.height = data_model.height
        self.strength = data_model.strength


class Engine(object):
    def __init__(self):
        pass

    def process(self, kwargs):
        return []


class EngineStableDiffusion(Engine):
    def __init__(self, pipe, config, sibling=None, custom_model_path=None, requires_safety_checker=True):
        super().__init__()
        if sibling is None:
#            self.engine = pipe.from_pretrained('runwayml/stable-diffusion-v1-5')
            self.engine = pipe.from_pretrained('runwayml/stable-diffusion-v1-5', use_auth_token=config.hf_token.strip())

        elif custom_model_path:
            if requires_safety_checker:
                self.engine = diffusers.StableDiffusionPipeline.from_pretrained(custom_model_path,
                                                                                safety_checker=sibling.engine.safety_checker,
                                                                                feature_extractor=sibling.engine.feature_extractor)
            else:
                self.engine = diffusers.StableDiffusionPipeline.from_pretrained(custom_model_path,
                                                                                feature_extractor=sibling.engine.feature_extractor)
        else:
            self.engine = pipe(
                vae=sibling.engine.vae,
                text_encoder=sibling.engine.text_encoder,
                tokenizer=sibling.engine.tokenizer,
                unet=sibling.engine.unet,
                scheduler=sibling.engine.scheduler,
                safety_checker=sibling.engine.safety_checker,
                feature_extractor=sibling.engine.feature_extractor
            )
        self.engine.to(is_cuda_available())

    def process(self, kwargs):
        output = self.engine(**kwargs)
        return {'image': output.images[0], 'nsfw': output.nsfw_content_detected[0]}


class EngineManager(object):
    def __init__(self):
        self.engines = {}

    def has_engine(self, name):
        return name in self.engines

    def add_engine(self, name, engine):
        if self.has_engine(name):
            return False
        self.engines[name] = engine
        return True

    def get_engine(self, name):
        if not self.has_engine(name):
            return None
        engine = self.engines[name]
        return engine


def get_img_config():
    try:
        config_file = open('./app/configs/image_token_config.json', 'r')
        config_data = json.loads(config_file.read())
        return ImgConfig(hf_token=config_data.get('hf_token'), custom_models=config_data.get('custom_models', []))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="config.json not found.")


def get_manager(config: ImgConfig = Depends(get_img_config)):
    config = get_img_config()
    manager = EngineManager()
    manager.add_engine('txt2img', EngineStableDiffusion(diffusers.StableDiffusionPipeline, config, sibling=None))
    manager.add_engine('img2img', EngineStableDiffusion(diffusers.StableDiffusionImg2ImgPipeline,
                                                        config, sibling=manager.get_engine('txt2img')))
    manager.add_engine('masking', EngineStableDiffusion(diffusers.StableDiffusionInpaintPipeline,
                                                        config, sibling=manager.get_engine('txt2img')))
    for custom_model in config.custom_models:
        manager.add_engine(custom_model['url_path'],
                           EngineStableDiffusion(diffusers.StableDiffusionPipeline, config,
                                                 sibling=manager.get_engine('txt2img'),
                                                 custom_model_path=custom_model['model_path'],
                                                 requires_safety_checker=custom_model['requires_safety_checker']))
    return manager


def pil_to_b64(pil_input):
    buffer = BytesIO()
    pil_input.save(buffer, 'PNG')
    output = base64.b64encode(buffer.getvalue()).decode('utf-8').replace('\n', '')
    buffer.close()
    return output


def b64_to_pil(b64_input):
    output = Image.open(BytesIO(base64.b64decode(b64_input)))
    return output


def is_cuda_available():
    try:
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'


def get_decoded_images(image_dict):
    decoded_images = []
    if "images" in image_dict:
        for image_data in image_dict["images"]:
            base64_data = image_data.get("base64", "")
            decoded_image = b64_to_pil(base64_data)
            decoded_images.append(decoded_image)

    return decoded_images
