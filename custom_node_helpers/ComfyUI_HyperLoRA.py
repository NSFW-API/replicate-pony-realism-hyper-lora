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
        # The correct URL for HyperLoRA models
        base_hyperlora_url = "https://huggingface.co/bytedance-research/HyperLoRA/resolve/main"

        return {
            "sdxl_hyper_id_lora_v1_fidelity/hyper_lora_modules.json": {
                "url": f"{base_hyperlora_url}/sdxl_hyper_id_lora_v1_fidelity/hyper_lora_modules.json",
                "dest": "ComfyUI/models/hyper_lora/hyper_lora/sdxl_hyper_id_lora_v1_fidelity/",
            },
            "sdxl_hyper_id_lora_v1_fidelity/hyper_lora_modules.safetensors": {
                "url": f"{base_hyperlora_url}/sdxl_hyper_id_lora_v1_fidelity/hyper_lora_modules.safetensors",
                "dest": "ComfyUI/models/hyper_lora/hyper_lora/sdxl_hyper_id_lora_v1_fidelity/",
            },
            "sdxl_hyper_id_lora_v1_fidelity/id_projector.safetensors": {
                "url": f"{base_hyperlora_url}/sdxl_hyper_id_lora_v1_fidelity/id_projector.safetensors",
                "dest": "ComfyUI/models/hyper_lora/hyper_lora/sdxl_hyper_id_lora_v1_fidelity/",
            },
            "sdxl_hyper_id_lora_v1_fidelity/resampler.safetensors": {
                "url": f"{base_hyperlora_url}/sdxl_hyper_id_lora_v1_fidelity/resampler.safetensors",
                "dest": "ComfyUI/models/hyper_lora/hyper_lora/sdxl_hyper_id_lora_v1_fidelity/",
            }
        }