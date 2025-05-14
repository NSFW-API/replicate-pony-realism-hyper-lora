import json
import os
import subprocess
import time
import shutil
from typing import List

from cog import BasePredictor, Input, Path

from comfyui import ComfyUI  # pip-installed via cog.yaml

# ---- constants -------------------------------------------------------------
WORKFLOW_JSON = "pony_realism_hyperlora_workflow.json"
OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
ALL_DIRS = [OUTPUT_DIR, INPUT_DIR]


# ----------------------------------------------------------------------------

class Predictor(BasePredictor):
    def setup(self):
        # Create necessary directories
        for dir_path in ALL_DIRS:
            os.makedirs(dir_path, exist_ok=True)

        # Clone HyperLoRA nodes if missing
        if not os.path.exists("ComfyUI/custom_nodes/ComfyUI-HyperLoRA"):
            print("Cloning ComfyUI-HyperLoRA custom node...")
            subprocess.check_call([
                "git", "clone", "--depth", "1",
                "https://github.com/bytedance/ComfyUI-HyperLoRA.git",
                "ComfyUI/custom_nodes/ComfyUI-HyperLoRA"
            ])

        if not os.path.exists("ComfyUI/custom_nodes/ComfyUI_ADV_CLIP_emb"):
            print("Cloning ComfyUI_ADV_CLIP_emb custom node...")
            subprocess.check_call([
                "git", "clone", "--depth", "1",
                "https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb.git",
                "ComfyUI/custom_nodes/ComfyUI_ADV_CLIP_emb"
            ])

        if not os.path.exists("ComfyUI/custom_nodes/ComfyUI-Impact-Pack"):
            print("Cloning ComfyUI-Impact-Pack custom node...")
            subprocess.check_call([
                "git", "clone", "--depth", "1",
                "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
                "ComfyUI/custom_nodes/ComfyUI-Impact-Pack"
            ])

        # Create model directories
        models_dir = "ComfyUI/models"
        os.makedirs(models_dir, exist_ok=True)

        # Create insightface directory
        insightface_dir = f"{models_dir}/insightface/models/antelopev2"
        os.makedirs(insightface_dir, exist_ok=True)

        # Create HyperLoRA directories
        hyperlora_base_dir = f"{models_dir}/hyper_lora"
        os.makedirs(hyperlora_base_dir, exist_ok=True)

        # Create clip processor directory
        clip_processor_dir = f"{hyperlora_base_dir}/clip_processor/clip_vit_large_14_processor"
        os.makedirs(clip_processor_dir, exist_ok=True)

        # Create clip vit directory
        clip_vit_dir = f"{hyperlora_base_dir}/clip_vit/clip_vit_large_14"
        os.makedirs(clip_vit_dir, exist_ok=True)

        # Create HyperLoRA model directory
        hyperlora_model_dir = f"{hyperlora_base_dir}/hyper_lora/sdxl_hyper_id_lora_v1_fidelity"
        os.makedirs(hyperlora_model_dir, exist_ok=True)

        # Download AntelopeV2 face detection model files
        antelopev2_files = {
            "2d106det.onnx": "https://huggingface.co/MonsterMMORPG/tools/resolve/main/2d106det.onnx",
            "1k3d68.onnx": "https://huggingface.co/MonsterMMORPG/tools/resolve/main/1k3d68.onnx",
            "genderage.onnx": "https://huggingface.co/MonsterMMORPG/tools/resolve/main/genderage.onnx",
            "glintr100.onnx": "https://huggingface.co/MonsterMMORPG/tools/resolve/main/glintr100.onnx"
        }

        for filename, url in antelopev2_files.items():
            model_path = f"{insightface_dir}/{filename}"
            if not os.path.exists(model_path):
                print(f"Downloading {filename} to {model_path}...")
                subprocess.check_call(["pget", "-vf", url, model_path])

        # Download HyperLoRA specific models

        # Download CLIP processor config
        clip_processor_config = f"{clip_processor_dir}/preprocessor_config.json"
        if not os.path.exists(clip_processor_config):
            print(f"Downloading CLIP processor config to {clip_processor_config}...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/preprocessor_config.json",
                clip_processor_config
            ])

        # Download CLIP ViT model files
        clip_vit_config = f"{clip_vit_dir}/config.json"
        if not os.path.exists(clip_vit_config):
            print(f"Downloading CLIP ViT config to {clip_vit_config}...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/config.json",
                clip_vit_config
            ])

        clip_vit_model = f"{clip_vit_dir}/model.safetensors"
        if not os.path.exists(clip_vit_model):
            print(f"Downloading CLIP ViT model to {clip_vit_model}...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/h94/IP-Adapter/resolve/main/models/clip_vit_large_patch14.safetensors",
                clip_vit_model
            ])

        # Download HyperLoRA model files
        base_hyperlora_url = "https://huggingface.co/bytedance-research/HyperLoRA/resolve/main"
        hyperlora_model_files = {
            "hyper_lora_modules.json": f"{base_hyperlora_url}/sdxl_hyper_id_lora_v1_fidelity/hyper_lora_modules.json",
            "hyper_lora_modules.safetensors": f"{base_hyperlora_url}/sdxl_hyper_id_lora_v1_fidelity/hyper_lora_modules.safetensors",
            "id_projector.safetensors": f"{base_hyperlora_url}/sdxl_hyper_id_lora_v1_fidelity/id_projector.safetensors",
            "resampler.safetensors": f"{base_hyperlora_url}/sdxl_hyper_id_lora_v1_fidelity/resampler.safetensors"
        }

        for filename, url in hyperlora_model_files.items():
            model_path = f"{hyperlora_model_dir}/{filename}"
            if not os.path.exists(model_path):
                print(f"Downloading {filename} to {model_path}...")
                subprocess.check_call(["pget", "-vf", url, model_path])

        # Download SDXL VAE
        vae_dir = f"{models_dir}/vae"
        os.makedirs(vae_dir, exist_ok=True)

        vae_path = f"{vae_dir}/sdxl_vae.safetensors"
        if not os.path.exists(vae_path):
            print(f"Downloading SDXL VAE to {vae_path}...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
                vae_path
            ])

        # Download T5 clip for SDXL
        clip_dir = f"{models_dir}/clip"
        os.makedirs(clip_dir, exist_ok=True)

        clip_path = f"{clip_dir}/t5xxl_fp16.safetensors"
        if not os.path.exists(clip_path):
            print(f"Downloading CLIP to {clip_path}...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors",
                clip_path
            ])

        # Start ComfyUI server
        print("Starting ComfyUI server...")
        self.comfy = ComfyUI("127.0.0.1:8188")
        self.comfy.start_server(OUTPUT_DIR, INPUT_DIR)

        # Wait for server to be ready
        max_retries = 30
        retry_count = 0
        while retry_count < max_retries:
            try:
                self.comfy.connect()
                print("ComfyUI server is ready")
                break
            except Exception as e:
                print(f"Waiting for ComfyUI server to be ready... {e}")
                time.sleep(1)
                retry_count += 1

        if retry_count == max_retries:
            raise RuntimeError("Failed to connect to ComfyUI server")

    # ---- helpers -----------------------------------------------------------
    def _nearest_multiple(self, x: int, k: int = 8) -> int:
        return ((x + k - 1) // k) * k

    # -----------------------------------------------------------------------

    def predict(
            self,
            prompt: str = Input(description="Main text prompt."),
            negative_prompt: str = Input(default="", description="Optional negative."),
            reference_image: Path = Input(
                description="An image containing a face that you want to use as reference for face swapping.",
            ),
            width: int = Input(default=768, ge=64, le=1536),
            height: int = Input(default=768, ge=64, le=1536),
            steps: int = Input(default=30, ge=1, le=150),
            cfg: float = Input(default=7.0, ge=1.0, le=20.0),
            sampler_name: str = Input(default="dpmpp_2m_sde", choices=["euler", "euler_ancestral", "heun", "dpmpp_2s_ancestral", "dpmpp_2m_sde"]),
            scheduler: str = Input(default="karras", choices=["karras", "normal"]),
            seed: int = Input(default=0, description="0 = random"),
            face_weight: float = Input(
                default=0.8,
                ge=0.0,
                le=1.0,
                description="Weight of the face adaptation effect (0.0 to 1.0)",
            )
    ) -> List[Path]:

        # 1. housekeeping
        self.comfy.cleanup(ALL_DIRS)
        if seed == 0:
            seed = int.from_bytes(os.urandom(2), "big")

        # Copy reference image to input directory
        if reference_image:
            reference_path = os.path.join(INPUT_DIR, "reference.png")
            shutil.copy2(reference_image, reference_path)
        else:
            # If no reference image, we can't use HyperLoRA effectively
            raise ValueError("A reference face image is required for HyperLoRA to work properly.")

        # 2. Load workflow.json
        with open(WORKFLOW_JSON) as f:
            wf = json.load(f)

        # 3. Update workflow with user inputs
        if "nodes" in wf:  # Style B
            by_id = {str(n["id"]): n for n in wf["nodes"]}
        else:  # Style A
            by_id = wf

        def node(idx: int):
            """Return the node dict for a given numeric id"""
            return by_id[str(idx)]

        # ----- prompt nodes -------------------------------------------------
        if str(4) in by_id:  # Prompt node
            node(4)["inputs"]["text"] = f"fcsks fxhks fhyks, {prompt}"

        if str(5) in by_id:  # Negative prompt node
            node(5)["inputs"]["text"] = negative_prompt

        # ----- latent size --------------------------------------------------
        if str(6) in by_id:  # Empty Latent Image node
            latent_inputs = node(6)["inputs"]
            latent_inputs["width"] = self._nearest_multiple(width)
            latent_inputs["height"] = self._nearest_multiple(height)
            latent_inputs["batch_size"] = 1

        # ----- sampler settings --------------------------------------------
        if str(7) in by_id:  # KSampler node
            sampler_inputs = node(7)["inputs"]
            sampler_inputs["seed"] = seed
            sampler_inputs["steps"] = steps
            sampler_inputs["cfg"] = cfg
            sampler_inputs["sampler_name"] = sampler_name
            sampler_inputs["scheduler"] = scheduler
            sampler_inputs["denoise"] = 1.0

        # ----- Face adapter settings ---------------------------------------
        # Update the HyperLoRAApplyLoRA node with weight if it exists
        if str(9) in by_id and "widgets_values" in by_id[str(9)]:
            by_id[str(9)]["widgets_values"][0] = face_weight

        # Make sure the ImageLoad node points to the reference image
        if str(17) in by_id:
            node(17)["inputs"]["image"] = "reference.png"

        # 4. Run the workflow
        print("Loading workflow...")
        wf_loaded = self.comfy.load_workflow(wf)
        print("Running workflow...")
        self.comfy.run_workflow(wf_loaded)

        # 5. Get the output images
        print("Getting output files...")
        all_files = self.comfy.get_files(OUTPUT_DIR)
        image_files = [
            p
            for p in all_files
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]
        return image_files