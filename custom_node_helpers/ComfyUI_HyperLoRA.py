from custom_node_helper import CustomNodeHelper

class ComfyUI_HyperLoRA(CustomNodeHelper):
    @staticmethod
    def models():
        return [
            "sdxl_hyper_id_lora_v1_fidelity/hyper_lora_modules.json",
            "sdxl_hyper_id_lora_v1_fidelity/hyper_lora_modules.safetensors",
            "sdxl_hyper_id_lora_v1_fidelity/id_projector.safetensors",
            "sdxl_hyper_id_lora_v1_fidelity/resampler.safetensors"
        ]

    @staticmethod
    def weights_map(base_url):
        # Define weights map for HyperLoRA models
        # These URLs would need to be the actual download locations for the model files
        return {
            "sdxl_hyper_id_lora_v1_fidelity/hyper_lora_modules.json": {
                "url": "https://huggingface.co/ByteDance/ComfyUI-HyperLoRA/resolve/main/sdxl_hyper_id_lora_v1_fidelity/hyper_lora_modules.json",
                "dest": "ComfyUI/models/hyper_lora/hyper_lora/sdxl_hyper_id_lora_v1_fidelity/",
            },
            "sdxl_hyper_id_lora_v1_fidelity/hyper_lora_modules.safetensors": {
                "url": "https://huggingface.co/ByteDance/ComfyUI-HyperLoRA/resolve/main/sdxl_hyper_id_lora_v1_fidelity/hyper_lora_modules.safetensors",
                "dest": "ComfyUI/models/hyper_lora/hyper_lora/sdxl_hyper_id_lora_v1_fidelity/",
            },
            "sdxl_hyper_id_lora_v1_fidelity/id_projector.safetensors": {
                "url": "https://huggingface.co/ByteDance/ComfyUI-HyperLoRA/resolve/main/sdxl_hyper_id_lora_v1_fidelity/id_projector.safetensors",
                "dest": "ComfyUI/models/hyper_lora/hyper_lora/sdxl_hyper_id_lora_v1_fidelity/",
            },
            "sdxl_hyper_id_lora_v1_fidelity/resampler.safetensors": {
                "url": "https://huggingface.co/ByteDance/ComfyUI-HyperLoRA/resolve/main/sdxl_hyper_id_lora_v1_fidelity/resampler.safetensors",
                "dest": "ComfyUI/models/hyper_lora/hyper_lora/sdxl_hyper_id_lora_v1_fidelity/",
            }
        }