### Using `egl` to make pyrender work
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import logging
from pathlib import Path
from typing import Tuple

from omegaconf import DictConfig, open_dict
import hydra
import gradio as gr
import trimesh

from utils import pytorch_utils, hydra_utils


###------ Utilities -------###
def convert_to_glb(mesh_path: Path, glb_path: Path) -> None:
    """Convert a mesh file to GLB for display in Gradio Model3D
    Args:
        mesh_path (_type_): Path to the source mesh file. Defaults to __None__
        glb_path (_type_): Path to the output .glb file. Defaults to __None__
    """
    ### Convert and enforce color
    mesh = trimesh.load_mesh(mesh_path)
    mesh.visual.vertex_colors = [128, 174, 128, 255]
    mesh.visual.face_colors = [128, 174, 128, 255]
    mesh.fix_normals()
    mesh.export(glb_path)


###------ Gradio App (HyperDiffusion Inference) -------###
@hydra.main(version_base=None, config_path="configs", config_name="train_hd_3D")
def main(cfg: DictConfig) -> None:
    logger = logging.getLogger(__name__)

    ### Pre-process configuration
    cfg = hydra_utils.preprocess_cfg(cfg, extra_folders=cfg.extra_folders, logger=logger, verbose=True)

    ### Create random generator collection
    rng = pytorch_utils.SeedAll(cfg.seed, logger=logger)

    ### Create Trainer (HyperDiffusion)
    hd_trainer = hydra.utils.instantiate(cfg.diffusion_method[cfg.diffusion_method.selected].trainer, cfg, logger, rng)

    ### Prepare output directory for Gradio artifacts
    gradio_dir = cfg.current_path / "gradio_hd_3D"
    gradio_dir.mkdir(parents=True, exist_ok=True)

    ### Handlers
    def generate_one() -> Tuple[str, str]:
        """Sample one mesh and return .glb path for Model3D
        Args:
            input_dims (_type_): _description_. Defaults to __None__
        """
        ### Sample weights
        x_0s = hd_trainer.sample_loop(hd_trainer.model, (1, hd_trainer.n_weights))  ### [1, n_weights]

        ### Export .ply
        export_path = gradio_dir / "ply"
        meshes, _, _ = hd_trainer.generate_meshes(x_0s, export_path=export_path, export_flag=True, return_occ_cubes=False, return_sample_points=False, stop_at_threshold=1)

        ### Convert to .glb for Gradio viewer
        ply_path = export_path / "mesh_0.ply"
        glb_path = gradio_dir / "mesh_0.glb"
        convert_to_glb(ply_path, glb_path)

        status = f"Generated one liver mesh at {glb_path=}"
        return str(glb_path), status

    ### Build UI
    with gr.Blocks() as demo:
        gr.Markdown("# HyperDiffusion 3D Liver - Inference Demo")
        with gr.Row():
            model3d = gr.Model3D(
                value=None,
                label="3D Liver",
                clear_color=[1.0, 1.0, 1.0, 1.0],
                scale=4,
                display_mode="solid",
                camera_position=(-100, 0, 1)
            )
            with gr.Column():
                btn_gen = gr.Button("Generate")
                status = gr.Textbox(value="", label="Status", show_label=True, interactive=False)

        btn_gen.click(fn=generate_one, inputs=[], outputs=[model3d, status])

    demo.launch(inline=False, server_port=7860, share=False)


if __name__ == "__main__":
    ### Close all existed gradio
    gr.close_all()
    main()