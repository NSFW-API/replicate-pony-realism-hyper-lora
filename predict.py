import json
import os
import shutil
import subprocess
import time
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

        # Create insightface directory with proper structure
        insightface_models_dir = f"{models_dir}/insightface/models"
        insightface_dir = f"{insightface_models_dir}/antelopev2"
        detection_dir = f"{insightface_dir}/detection"
        os.makedirs(insightface_models_dir, exist_ok=True)
        os.makedirs(insightface_dir, exist_ok=True)
        os.makedirs(detection_dir, exist_ok=True)

        # Create InstantID directories
        instantid_dir = f"{models_dir}/instantid"
        os.makedirs(instantid_dir, exist_ok=True)
        controlnet_dir = f"{models_dir}/controlnet"
        os.makedirs(controlnet_dir, exist_ok=True)

        # Create FaceDetailer directories (from Impact Pack)
        ultralytics_dir = f"{models_dir}/ultralytics"
        os.makedirs(ultralytics_dir, exist_ok=True)
        face_detection_dir = f"{ultralytics_dir}/bbox"
        os.makedirs(face_detection_dir, exist_ok=True)

        # Other model directories
        vae_dir = f"{models_dir}/vae"
        checkpoints_dir = f"{models_dir}/checkpoints"
        clip_dir = f"{models_dir}/clip"
        os.makedirs(vae_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(clip_dir, exist_ok=True)

        # Download all required models
        self._download_models(
            models_dir=models_dir,
            insightface_dir=insightface_dir,
            insightface_models_dir=insightface_models_dir,
            instantid_dir=instantid_dir,
            controlnet_dir=controlnet_dir,
            face_detection_dir=face_detection_dir
        )

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

        # Clone a-person-mask-generator repository
        if not os.path.exists("ComfyUI/custom_nodes/a-person-mask-generator"):
            print("Cloning a-person-mask-generator custom node...")
            subprocess.check_call([
                "git", "clone", "--depth", "1",
                "https://github.com/djbielejeski/a-person-mask-generator.git",
                "ComfyUI/custom_nodes/a-person-mask-generator"
            ])

            # Install a-person-mask-generator requirements directly
            print("Installing a-person-mask-generator dependencies...")
            subprocess.check_call([
                "pip", "install", "mediapipe==0.10.11"
            ])

        # Clone ComfyUI-Impact-Pack repository for the GrowMask node and FaceDetailer
        if not os.path.exists("ComfyUI/custom_nodes/ComfyUI-Impact-Pack"):
            print("Cloning ComfyUI-Impact-Pack custom node...")
            subprocess.check_call([
                "git", "clone", "--depth", "1",
                "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
                "ComfyUI/custom_nodes/ComfyUI-Impact-Pack"
            ])

            # Install Impact Pack dependencies directly
            print("Installing ComfyUI-Impact-Pack dependencies...")
            subprocess.check_call([
                "pip", "install", "piexif==1.1.3", "wget==3.2", "ultralytics==8.0.145"
            ])

        # Clone ComfyUI_InstantID repository
        if not os.path.exists("ComfyUI/custom_nodes/ComfyUI_InstantID"):
            print("Cloning ComfyUI_InstantID custom node...")
            subprocess.check_call([
                "git", "clone", "--depth", "1",
                "https://github.com/cubiq/ComfyUI_InstantID.git",
                "ComfyUI/custom_nodes/ComfyUI_InstantID"
            ])

            # Install ComfyUI_InstantID dependencies
            print("Installing ComfyUI_InstantID dependencies...")
            subprocess.check_call([
                "pip", "install", "insightface==0.7.3", "onnxruntime==1.16.3"
            ])

        # Add this new code to clone Impact Subpack
        if not os.path.exists("ComfyUI/custom_nodes/ComfyUI-Impact-Subpack"):
            print("Cloning ComfyUI-Impact-Subpack custom node...")
            subprocess.check_call([
                "git", "clone", "--depth", "1",
                "https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git",
                "ComfyUI/custom_nodes/ComfyUI-Impact-Subpack"
            ])

    def _download_models(self, models_dir, insightface_dir, insightface_models_dir,
                         instantid_dir, controlnet_dir, face_detection_dir):
        """Download all required model files"""
        # 1. CLIP processor config
        clip_processor_config = f"{models_dir}/hyper_lora/clip_processor/clip_vit_large_14_processor/preprocessor_config.json"
        if not os.path.exists(clip_processor_config):
            print(f"Downloading CLIP processor config...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/preprocessor_config.json",
                clip_processor_config
            ])

        # 2. CLIP ViT config and model
        clip_vit_config = f"{models_dir}/hyper_lora/clip_vit/clip_vit_large_14/config.json"
        if not os.path.exists(clip_vit_config):
            print(f"Downloading CLIP ViT config...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/config.json",
                clip_vit_config
            ])

        clip_vit_model = f"{models_dir}/hyper_lora/clip_vit/clip_vit_large_14/model.safetensors"
        if not os.path.exists(clip_vit_model):
            print(f"Downloading CLIP ViT model...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors",
                clip_vit_model
            ])

        # 3. HyperLoRA model files
        hyperlora_base_dir = f"{models_dir}/hyper_lora/hyper_lora/sdxl_hyper_id_lora_v1_fidelity"
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

        for filename, url in antelopev2_files.items():
            filepath = os.path.join(insightface_dir, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                subprocess.check_call(["pget", "-vf", url, filepath])

        # 5. Download AntelopeV2 zip with detection models
        if not os.path.exists(f"{insightface_dir}/detection/model-0000.params"):
            print("Downloading AntelopeV2 models...")
            # Since AntelopeV2 comes as a zip with multiple files, we need to download and extract it
            tmp_zip = "/tmp/antelopev2.zip"
            subprocess.check_call([
                "pget", "-vf",
                "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
                tmp_zip
            ])
            # Extract the zip file
            subprocess.check_call(["unzip", "-o", tmp_zip, "-d", insightface_models_dir])

            # Fix directory structure - move files one level up if nested
            if os.path.exists(f"{insightface_dir}/antelopev2"):
                print("Fixing nested directory structure...")
                # Move all files from nested directory up one level
                os.system(f"mv {insightface_dir}/antelopev2/* {insightface_dir}/")
                # Remove the now-empty directory
                os.system(f"rmdir {insightface_dir}/antelopev2 || true")

            # Clean up
            os.remove(tmp_zip)

            # Debug check
            print(f"Listing AntelopeV2 directory contents: {insightface_dir}")
            os.system(f"ls -la {insightface_dir}")
            print(f"Checking for detection directory: {insightface_dir}/detection")
            os.system(f"ls -la {insightface_dir}/detection || echo 'Detection directory not found'")

        # 6. SDXL VAE
        vae_path = f"{models_dir}/vae/sdxl_vae.safetensors"
        if not os.path.exists(vae_path):
            print(f"Downloading SDXL VAE...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
                vae_path
            ])

        # 7. RealVis checkpoint
        checkpoint_path = f"{models_dir}/checkpoints/realvis.safetensors"
        if not os.path.exists(checkpoint_path):
            print(f"Downloading RealVis checkpoint...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/NSFW-API/RealVisXL/resolve/main/realvisXL_v5.safetensors",
                checkpoint_path
            ])

        # 8. T5 CLIP for SDXL (if needed)
        clip_path = f"{models_dir}/clip/t5xxl_fp16.safetensors"
        if not os.path.exists(clip_path):
            print(f"Downloading T5 CLIP...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors",
                clip_path
            ])

        # 9. InstantID ControlNet and IP-Adapter model using tar files
        instantid_checkpoints_dir = f"{models_dir}/instantid"
        os.makedirs(instantid_checkpoints_dir, exist_ok=True)

        if not os.path.exists(f"{controlnet_dir}/instantid_controlnet.safetensors") or not os.path.exists(
                f"{instantid_dir}/ip-adapter.bin"):
            print(f"Downloading InstantID checkpoints (ControlNet and IP-Adapter)...")
            instantid_checkpoints_url = "https://weights.replicate.delivery/default/InstantID/checkpoints.tar"
            subprocess.check_call([
                "pget", "-vf", "-x",
                instantid_checkpoints_url,
                instantid_checkpoints_dir
            ])

            # Check for extracted files at their actual locations
            if os.path.exists(f"{instantid_dir}/ControlNetModel/diffusion_pytorch_model.safetensors"):
                # Copy to the expected location for the workflow
                shutil.copy2(
                    f"{instantid_dir}/ControlNetModel/diffusion_pytorch_model.safetensors",
                    f"{controlnet_dir}/instantid_controlnet.safetensors"
                )
                print(f"Copied InstantID ControlNet model to {controlnet_dir}/instantid_controlnet.safetensors")

            # No need to copy ip-adapter.bin since it's already in the right location

            # Verify files exist
            if not os.path.exists(f"{instantid_dir}/ip-adapter.bin"):
                raise RuntimeError(f"InstantID ip-adapter.bin not found after download!")
            if not os.path.exists(f"{controlnet_dir}/instantid_controlnet.safetensors"):
                raise RuntimeError(f"InstantID controlnet model not found after download!")

        # 10. InstantID IP-Adapter model
        instantid_ip_adapter_path = f"{instantid_dir}/ip-adapter.bin"
        if not os.path.exists(instantid_ip_adapter_path):
            print(f"Downloading InstantID IP-Adapter...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/InstantID/InstantID/resolve/main/ip-adapter.bin",
                instantid_ip_adapter_path
            ])

        # 11. FaceDetailer - YOLOv8 face detection model
        faceyolo_path = f"{face_detection_dir}/face_yolov8m.pt"
        if not os.path.exists(faceyolo_path):
            print(f"Downloading YOLOv8 face detection model...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt",
                faceyolo_path
            ])

    # ---- helpers -----------------------------------------------------------
    def _nearest_multiple(self, x: int, k: int = 8) -> int:
        return ((x + k - 1) // k) * k

    # -----------------------------------------------------------------------

    def predict(
            self,
            prompt: str = Input(description="Main text prompt."),
            negative_prompt: str = Input(default="lowres, bad anatomy", description="Negative prompt."),
            source_image: Path = Input(description="The base image where the face will be replaced"),
            reference_image: Path = Input(
                description="An image containing a face that you want to use as reference for face swapping.",
                default=None,
            ),
            width: int = Input(default=768, ge=64, le=1536),
            height: int = Input(default=1024, ge=64, le=1536),
            steps: int = Input(default=30, ge=1, le=150),
            cfg: float = Input(default=7.0, ge=1.0, le=20.0),
            sampler_name: str = Input(default="dpmpp_2m_sde",
                                      choices=["euler", "euler_ancestral", "heun", "dpmpp_2s_ancestral", "dpmpp_2m",
                                               "dpmpp_2m_sde", "dpmpp_sde", "uni_pc"]),
            scheduler: str = Input(default="karras", choices=["normal", "karras"]),
            seed: int = Input(default=0, description="0 = random"),
            face_weight: float = Input(
                default=0.8,
                ge=0.0,
                le=1.0,
                description="Weight of the face adaptation effect (0.0 to 1.0)",
            ),
            instantid_weight: float = Input(
                default=0.8,
                ge=0.0,
                le=1.0,
                description="Weight of the InstantID effect (0.0 to 1.0)",
            ),
            facedetail_strength: float = Input(
                default=0.5,
                ge=0.0,
                le=1.0,
                description="Strength of face detail enhancement",
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

        source_path = os.path.join(INPUT_DIR, "source.png")
        shutil.copy2(source_image, source_path)

        # 2. Load workflow with combined technologies
        print("Loading combined HyperLoRA + InstantID + FaceDetailer workflow")

        with open(WORKFLOW_JSON) as f:
            wf = json.load(f)

        # 3. Update workflow with user inputs
        if "nodes" in wf:  # Style B
            by_id = {str(n["id"]): n for n in wf["nodes"]}
        else:  # Style A
            by_id = wf

        def node(idx):
            """Return the node dict for a given numeric id"""
            return by_id[str(idx)]

        # ----- prompt nodes and images -------------------------------------------------
        node(4)["inputs"]["text"] = f"fcsks fxhks fhyks, {prompt}"
        node(5)["inputs"]["text"] = negative_prompt

        # Make sure reference and source images are set
        if "17" in by_id:  # Reference image
            node(17)["inputs"]["image"] = "reference.png"
        if "20" in by_id:  # Source image
            node(20)["inputs"]["image"] = "source.png"

        # ----- latent size --------------------------------------------------
        if "6" in by_id:
            node(6)["inputs"]["width"] = self._nearest_multiple(width)
            node(6)["inputs"]["height"] = self._nearest_multiple(height)
            node(6)["inputs"]["batch_size"] = 1

        # ----- sampler settings --------------------------------------------
        if "7" in by_id:
            node(7)["inputs"]["seed"] = seed
            node(7)["inputs"]["steps"] = steps
            node(7)["inputs"]["cfg"] = cfg
            node(7)["inputs"]["sampler_name"] = sampler_name
            node(7)["inputs"]["scheduler"] = scheduler

        # ----- HyperLoRA settings ------------------------------------------
        if "9h" in by_id:
            node("9h")["inputs"]["weight"] = face_weight

        # ----- InstantID settings ------------------------------------------
        if "33" in by_id:
            node(33)["inputs"]["weight_faceid"] = instantid_weight
            node(33)["inputs"]["weight_cnet"] = instantid_weight

        # ----- FaceDetailer settings ---------------------------------------
        if "50" in by_id:
            node(50)["inputs"]["denoise"] = facedetail_strength
            node(50)["inputs"]["steps"] = int(steps * 0.75)  # Use fewer steps for detailing
            node(50)["inputs"]["cfg"] = cfg
            node(50)["inputs"]["seed"] = seed + 1  # Use a different seed for variation

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
