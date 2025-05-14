import json
import os
import subprocess
import time
import shutil
from typing import List

from cog import BasePredictor, Input, Path

from comfyui import ComfyUI  # pip-installed via cog.yaml

# ---- constants -------------------------------------------------------------
WORKFLOW_JSON = "hyperlora_workflow.json"
OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
ALL_DIRS = [OUTPUT_DIR, INPUT_DIR]

# ----------------------------------------------------------------------------

class Predictor(BasePredictor):
    def setup(self):
        # Create necessary directories
        for dir_path in ALL_DIRS:
            os.makedirs(dir_path, exist_ok=True)

        # Clone required custom nodes
        self._clone_custom_nodes()

        # Create model directories
        models_dir = "ComfyUI/models"
        os.makedirs(models_dir, exist_ok=True)

        # Create HyperLoRA directories
        hyper_lora_dir = f"{models_dir}/hyper_lora"
        clip_processor_dir = f"{hyper_lora_dir}/clip_processor/clip_vit_large_14_processor"
        clip_vit_dir = f"{hyper_lora_dir}/clip_vit/clip_vit_large_14"
        hyperlora_fidelity_dir = f"{hyper_lora_dir}/hyper_lora/sdxl_hyper_id_lora_v1_fidelity"

        os.makedirs(clip_processor_dir, exist_ok=True)
        os.makedirs(clip_vit_dir, exist_ok=True)
        os.makedirs(hyperlora_fidelity_dir, exist_ok=True)

        # Create insightface directory with correct structure
        insightface_models_dir = f"{models_dir}/insightface/models"
        antelopev2_dir = f"{insightface_models_dir}/antelopev2"
        detection_dir = f"{antelopev2_dir}/detection"
        os.makedirs(insightface_models_dir, exist_ok=True)
        os.makedirs(antelopev2_dir, exist_ok=True)
        os.makedirs(detection_dir, exist_ok=True)

        # Other model directories
        vae_dir = f"{models_dir}/vae"
        checkpoints_dir = f"{models_dir}/checkpoints"
        clip_dir = f"{models_dir}/clip"
        os.makedirs(vae_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(clip_dir, exist_ok=True)

        # Download all required models
        self._download_models()

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

    def _clone_custom_nodes(self):
        """Clone required custom node repositories"""
        # Clone HyperLoRA repository
        if not os.path.exists("ComfyUI/custom_nodes/ComfyUI-HyperLoRA"):
            print("Cloning ComfyUI-HyperLoRA custom node...")
            subprocess.check_call([
                "git", "clone", "--depth", "1",
                "https://github.com/bytedance/ComfyUI-HyperLoRA.git",
                "ComfyUI/custom_nodes/ComfyUI-HyperLoRA"
            ])

    def _download_models(self):
        """Download all required model files"""
        # 1. CLIP processor config
        clip_processor_config = "ComfyUI/models/hyper_lora/clip_processor/clip_vit_large_14_processor/preprocessor_config.json"
        if not os.path.exists(clip_processor_config):
            print(f"Downloading CLIP processor config...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/preprocessor_config.json",
                clip_processor_config
            ])

        # 2. CLIP ViT config and model
        clip_vit_config = "ComfyUI/models/hyper_lora/clip_vit/clip_vit_large_14/config.json"
        if not os.path.exists(clip_vit_config):
            print(f"Downloading CLIP ViT config...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/config.json",
                clip_vit_config
            ])

        clip_vit_model = "ComfyUI/models/hyper_lora/clip_vit/clip_vit_large_14/model.safetensors"
        if not os.path.exists(clip_vit_model):
            print(f"Downloading CLIP ViT model...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors",
                clip_vit_model
            ])

        # 3. HyperLoRA model files
        hyperlora_base_dir = "ComfyUI/models/hyper_lora/hyper_lora/sdxl_hyper_id_lora_v1_fidelity"
        hyperlora_base_url = "https://huggingface.co/bytedance-research/HyperLoRA/resolve/main/sdxl_hyper_id_lora_v1_fidelity"
        hyperlora_files = [
            "hyper_lora_modules.json",
            "hyper_lora_modules.safetensors",
            "id_projector.safetensors",
            "resampler.safetensors"
        ]

        for file in hyperlora_files:
            file_path = f"{hyperlora_base_dir}/{file}"
            if not os.path.exists(file_path):
                print(f"Downloading HyperLoRA {file}...")
                subprocess.check_call([
                    "pget", "-vf",
                    f"{hyperlora_base_url}/{file}",
                    file_path
                ])

        # 4. AntelopeV2 face model files
        antelopev2_files = {
            "2d106det.onnx": "https://huggingface.co/MonsterMMORPG/tools/resolve/main/2d106det.onnx",
            "1k3d68.onnx": "https://huggingface.co/MonsterMMORPG/tools/resolve/main/1k3d68.onnx",
            "genderage.onnx": "https://huggingface.co/MonsterMMORPG/tools/resolve/main/genderage.onnx",
            "glintr100.onnx": "https://huggingface.co/MonsterMMORPG/tools/resolve/main/glintr100.onnx"
        }

        insightface_dir = "ComfyUI/models/insightface/models/antelopev2"
        for filename, url in antelopev2_files.items():
            filepath = os.path.join(insightface_dir, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                subprocess.check_call(["pget", "-vf", url, filepath])

        # Download AntelopeV2 zip with all models including detection
        if not os.path.exists(f"{antelopev2_dir}/detection/model-0000.params"):
            print("Downloading AntelopeV2 models...")
            # Since AntelopeV2 comes as a zip with multiple files, we need to download and extract it
            tmp_zip = "/tmp/antelopev2.zip"
            subprocess.check_call([
                "pget", "-vf",
                "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
                tmp_zip
            ])
            # Extract the zip file
            subprocess.check_call(["unzip", "-o", tmp_zip, "-d", f"{models_dir}/insightface/models"])

            # Fix directory structure - move files one level up if nested
            if os.path.exists(f"{antelopev2_dir}/antelopev2"):
                print("Fixing nested directory structure...")
                # Move all files from nested directory up one level
                os.system(f"mv {antelopev2_dir}/antelopev2/* {antelopev2_dir}/")
                # Remove the now-empty directory
                os.system(f"rmdir {antelopev2_dir}/antelopev2 || true")

            # Clean up
            os.remove(tmp_zip)

            # Debug check
            print(f"Listing AntelopeV2 directory contents: {antelopev2_dir}")
            os.system(f"ls -la {antelopev2_dir}")
            print(f"Checking for detection directory: {antelopev2_dir}/detection")
            os.system(f"ls -la {antelopev2_dir}/detection || echo 'Detection directory not found'")

        # 5. SDXL VAE
        vae_path = "ComfyUI/models/vae/sdxl_vae.safetensors"
        if not os.path.exists(vae_path):
            print(f"Downloading SDXL VAE...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
                vae_path
            ])

        # 6. Pony Realism checkpoint
        checkpoint_path = "ComfyUI/models/checkpoints/pony_realism_23.safetensors"
        if not os.path.exists(checkpoint_path):
            print(f"Downloading Pony Realism checkpoint...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/NSFW-API/PonyRealism/resolve/main/pony_realism_23.safetensors",
                checkpoint_path
            ])

        # 7. T5 CLIP for SDXL (if needed)
        clip_path = "ComfyUI/models/clip/t5xxl_fp16.safetensors"
        if not os.path.exists(clip_path):
            print(f"Downloading T5 CLIP...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors",
                clip_path
            ])

    # ---- helpers -----------------------------------------------------------
    def _nearest_multiple(self, x: int, k: int = 8) -> int:
        return ((x + k - 1) // k) * k

    # -----------------------------------------------------------------------

    def predict(
            self,
            prompt: str = Input(description="Main text prompt."),
            negative_prompt: str = Input(default="lowres, bad anatomy", description="Negative prompt."),
            reference_image: Path = Input(
                description="An image containing a face that you want to use as reference for face swapping.",
                default=None,
            ),
            width: int = Input(default=768, ge=64, le=1536),
            height: int = Input(default=1024, ge=64, le=1536),
            steps: int = Input(default=30, ge=1, le=150),
            cfg: float = Input(default=7.0, ge=1.0, le=20.0),
            sampler_name: str = Input(default="dpmpp_2m_sde", choices=["euler", "euler_ancestral", "heun", "dpmpp_2s_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde", "uni_pc"]),
            scheduler: str = Input(default="karras", choices=["normal", "karras"]),
            seed: int = Input(default=0, description="0 = random"),
            face_weight: float = Input(
                default=0.8,
                ge=0.0,
                le=1.0,
                description="Weight of the face adaptation effect (0.0 to 1.0)",
            ),
    ) -> List[Path]:

        # 1. housekeeping
        self.comfy.cleanup(ALL_DIRS)
        if seed == 0: seed = int.from_bytes(os.urandom(2), "big")

        # If reference image is provided, copy it to input directory
        if reference_image:
            reference_path = os.path.join(INPUT_DIR, "reference.png")
            shutil.copy2(reference_image, reference_path)
        else:
            # HyperLoRA requires a reference image
            print("Error: HyperLoRA requires a reference face image to work properly.")
            raise ValueError("A reference image with a face is required.")

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
        node(4)["inputs"]["text"] = f"fcsks fxhks fhyks, {prompt}"
        node(5)["inputs"]["text"] = negative_prompt

        # ----- latent size --------------------------------------------------
        latent_inputs = node(6)["inputs"]
        latent_inputs["width"] = self._nearest_multiple(width)
        latent_inputs["height"] = self._nearest_multiple(height)
        latent_inputs["batch_size"] = 1

        # ----- sampler settings --------------------------------------------
        sampler_inputs = node(7)["inputs"]
        sampler_inputs["seed"] = seed
        sampler_inputs["steps"] = steps
        sampler_inputs["cfg"] = cfg
        sampler_inputs["sampler_name"] = sampler_name
        sampler_inputs["scheduler"] = scheduler
        sampler_inputs["denoise"] = 1.0

        # ----- HyperLoRA settings ------------------------------------------
        # Update the HyperLoRAApplyLoRA node with the user's face weight
        node(9)["inputs"]["weight"] = face_weight

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