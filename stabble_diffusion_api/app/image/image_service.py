from fastapi import HTTPException
from fastapi.responses import JSONResponse
import torch
from numba import cuda
from app.image.image_def import (TaskParams, EngineManager,
                                 is_cuda_available, pil_to_b64, b64_to_pil)


async def dream(task: str, params: TaskParams, manager: EngineManager):
    engine = manager.get_engine(task)
    output_data = {}

    try:
        total_results = []
        for i in range(params.num_outputs):
            new_seed = params.seed if params.seed == 0 else torch.Generator(
                device=is_cuda_available()).manual_seed(params.seed).seed()
            args_dict = {
                'prompt': [params.prompt] if params.prompt else None,
                'num_inference_steps': params.num_inference_steps,
                'guidance_scale': params.guidance_scale,
                'eta': params.eta,
                'generator': torch.Generator(device=is_cuda_available()),
            }
            if task == 'txt2img':
                args_dict['width'] = params.width
                args_dict['height'] = params.height

            if task == 'img2img' or task == 'masking':
                init_img_pil = b64_to_pil(params.init_image.file.read()) if params.init_image else None
                args_dict['init_image'] = init_img_pil
                args_dict['strength'] = params.strength

            if task == 'masking':
                mask_img_pil = b64_to_pil(params.mask_image.file.read()) if params.mask_image else None
                args_dict['mask_image'] = mask_img_pil

            pipeline_output = engine.process(args_dict)
            pipeline_output['seed'] = new_seed
            total_results.append(pipeline_output)

        output_data['status'] = 'success'
        images = []
        for result in total_results:
            images.append({
                'base64': pil_to_b64(result['image'].convert('RGB')),
                'seed': result['seed'],
                'mime_type': 'image/png',
                'nsfw': result['nsfw']
            })

        output_data['images'] = images

    except RuntimeError as e:
        output_data['status'] = 'failure'
        output_data['message'] = ('A RuntimeError occurred. You probably ran out of GPU memory. Check the server logs '
                                  'for more details.')
        cuda.get_current_device().reset()

        print(str(e))
        raise HTTPException(status_code=500, detail=output_data)

    return output_data
