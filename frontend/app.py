"""
SoloStudio-AI — Gradio frontend.

Tabs:
  1. Record / Upload  – capture or upload driver performance
  2. Character Gallery – manage character profiles (sketch + voice model)
  3. Render            – kick off a GPU render job and watch its status
  4. Outputs           – browse and download finished videos
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import httpx

API_BASE = "http://localhost:8000"
client = httpx.Client(base_url=API_BASE, timeout=60)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _api(method: str, path: str, **kwargs):
    resp = getattr(client, method)(path, **kwargs)
    resp.raise_for_status()
    return resp.json()


def _character_choices() -> list[tuple[str, str]]:
    """Returns list of (display_label, character_id) tuples."""
    chars = _api("get", "/characters")
    return [(c["name"], c["id"]) for c in chars]


# ── Tab 1: Record / Upload ─────────────────────────────────────────────────────

def upload_recording(video_file) -> str:
    if video_file is None:
        return "No file selected."
    with open(video_file, "rb") as f:
        resp = client.post(
            "/upload-recording",
            files={"file": (Path(video_file).name, f, "video/mp4")},
        )
    resp.raise_for_status()
    filename = resp.json()["filename"]
    return f"Uploaded: {filename}"


def build_record_tab() -> gr.Tab:
    with gr.Tab("Record / Upload") as tab:
        gr.Markdown("## Step 1 — Capture Your Performance")
        gr.Markdown(
            "Record yourself with your webcam or upload a pre-recorded clip. "
            "This is your **driver** performance — your face, body, and voice "
            "will be transferred onto your chosen character."
        )
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(
                    label="Webcam / Upload",
                    sources=["webcam", "upload"],
                    format="mp4",
                )
            with gr.Column():
                upload_btn = gr.Button("Upload to Server", variant="primary")
                upload_status = gr.Textbox(label="Status", interactive=False)

        upload_btn.click(upload_recording, inputs=video_input, outputs=upload_status)
    return tab


# ── Tab 2: Character Gallery ───────────────────────────────────────────────────

def create_character(name, notes, sketch, voice_model, voice_index, voice_sample):
    missing = [
        label
        for label, val in [
            ("Name", name),
            ("Sketch PNG", sketch),
            ("Voice Model (.pth)", voice_model),
            ("Voice Index (.index)", voice_index),
            ("Voice Sample (.wav)", voice_sample),
        ]
        if not val
    ]
    if missing:
        return f"Missing fields: {', '.join(missing)}", None, []

    with (
        open(sketch, "rb") as sk,
        open(voice_model, "rb") as vm,
        open(voice_index, "rb") as vi,
        open(voice_sample, "rb") as vs,
    ):
        resp = client.post(
            "/characters",
            data={"name": name, "notes": notes or ""},
            files={
                "sketch":       (Path(sketch).name,       sk, "image/png"),
                "voice_model":  (Path(voice_model).name,  vm, "application/octet-stream"),
                "voice_index":  (Path(voice_index).name,  vi, "application/octet-stream"),
                "voice_sample": (Path(voice_sample).name, vs, "audio/wav"),
            },
        )
    resp.raise_for_status()
    profile = resp.json()
    return f"Character '{profile['name']}' created (id={profile['id']})", None, refresh_gallery()


def delete_character(character_id: str):
    if not character_id:
        return "Select a character first.", refresh_gallery()
    client.delete(f"/characters/{character_id}").raise_for_status()
    return f"Deleted {character_id}", refresh_gallery()


def refresh_gallery():
    chars = _api("get", "/characters")
    rows = [[c["name"], c["id"], c.get("notes") or ""] for c in chars]
    return rows


def build_gallery_tab() -> gr.Tab:
    with gr.Tab("Character Gallery") as tab:
        gr.Markdown("## Character Profiles")
        gr.Markdown(
            "Each profile bundles a **sketch PNG** (the character's look) "
            "with an **RVC voice model** (the character's voice). "
            "Train the voice model separately using the voice sample."
        )
        with gr.Row():
            with gr.Column(scale=2):
                gallery_table = gr.Dataframe(
                    headers=["Name", "ID", "Notes"],
                    label="Saved Characters",
                    interactive=False,
                )
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    selected_id = gr.Textbox(label="Selected ID (paste from table)")
                    delete_btn = gr.Button("Delete Selected", variant="stop")
                delete_status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### Add New Character")
                char_name = gr.Textbox(label="Character Name")
                char_notes = gr.Textbox(label="Notes (optional)")
                sketch_file = gr.File(label="Character Sketch (.png)", file_types=[".png"])
                voice_model_file = gr.File(label="RVC Voice Model (.pth)", file_types=[".pth"])
                voice_index_file = gr.File(label="RVC Index File (.index)", file_types=[".index"])
                voice_sample_file = gr.File(label="Original Voice Sample (.wav)", file_types=[".wav"])
                create_btn = gr.Button("Save Character", variant="primary")
                create_status = gr.Textbox(label="Status", interactive=False)

        create_btn.click(
            create_character,
            inputs=[char_name, char_notes, sketch_file, voice_model_file,
                    voice_index_file, voice_sample_file],
            outputs=[create_status, sketch_file, gallery_table],
        )
        refresh_btn.click(refresh_gallery, outputs=gallery_table)
        delete_btn.click(
            delete_character, inputs=selected_id,
            outputs=[delete_status, gallery_table]
        )
        tab.select(refresh_gallery, outputs=gallery_table)
    return tab


# ── Tab 3: Render ──────────────────────────────────────────────────────────────

def start_render(recording_filename: str, character_id: str):
    if not recording_filename or not character_id:
        return "Provide both a recording filename and a character.", "", gr.update()
    data = _api(
        "post", "/render",
        json={"character_id": character_id, "recording_filename": recording_filename},
    )
    return f"Job started: {data['job_id']}", data["job_id"], gr.update()


def poll_status(job_id: str):
    if not job_id:
        return "No job running.", None
    data = _api("get", f"/render/{job_id}")
    status = data["status"]
    if status == "done":
        return f"Done! Output: {data['output_filename']}", data.get("output_filename")
    if status == "failed":
        return f"Failed: {data.get('error', 'unknown error')}", None
    return f"Status: {status} (auto-refreshing…)", None


def build_render_tab() -> gr.Tab:
    with gr.Tab("Render") as tab:
        gr.Markdown("## Step 3 — Render")
        gr.Markdown(
            "Choose your uploaded recording and the target character, then hit "
            "**Render**. SoloStudio will spin up an RTX 4090 pod, run voice "
            "conversion + motion transfer, pull back the result, and shut the "
            "pod down automatically."
        )
        with gr.Row():
            recording_name = gr.Textbox(
                label="Recording Filename",
                placeholder="myscene_take1.mp4",
            )
            char_dropdown = gr.Dropdown(
                label="Character",
                choices=[],
                allow_custom_value=False,
            )
            refresh_chars_btn = gr.Button("Refresh Characters")

        render_btn = gr.Button("Render", variant="primary", size="lg")
        render_status = gr.Textbox(label="Status", interactive=False)
        job_id_box = gr.Textbox(label="Job ID", interactive=False, visible=False)
        output_name_box = gr.Textbox(label="Output Filename", interactive=False)

        poll_btn = gr.Button("Check Status")
        auto_poll = gr.Timer(value=15, active=False)

        refresh_chars_btn.click(
            lambda: gr.update(choices=_character_choices()),
            outputs=char_dropdown,
        )
        render_btn.click(
            start_render,
            inputs=[recording_name, char_dropdown],
            outputs=[render_status, job_id_box, auto_poll],
        ).then(lambda: gr.update(active=True), outputs=auto_poll)

        poll_btn.click(poll_status, inputs=job_id_box, outputs=[render_status, output_name_box])
        auto_poll.tick(poll_status, inputs=job_id_box, outputs=[render_status, output_name_box])
        tab.select(
            lambda: gr.update(choices=_character_choices()), outputs=char_dropdown
        )
    return tab


# ── Tab 4: Outputs ─────────────────────────────────────────────────────────────

def build_outputs_tab() -> gr.Tab:
    with gr.Tab("Outputs") as tab:
        gr.Markdown("## Finished Videos")
        output_filename_input = gr.Textbox(
            label="Output Filename", placeholder="<job_id>_output.mp4"
        )
        with gr.Row():
            preview_btn = gr.Button("Preview")
            download_btn = gr.Button("Download Link")
        video_preview = gr.Video(label="Preview", interactive=False)
        download_link = gr.Textbox(label="Direct URL", interactive=False)

        def preview_output(filename: str):
            if not filename:
                return None
            return f"{API_BASE}/outputs/{filename}"

        def get_download_link(filename: str):
            if not filename:
                return ""
            return f"{API_BASE}/outputs/{filename}"

        preview_btn.click(preview_output, inputs=output_filename_input, outputs=video_preview)
        download_btn.click(get_download_link, inputs=output_filename_input, outputs=download_link)
    return tab


# ── App assembly ───────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="SoloStudio-AI",
        theme=gr.themes.Soft(primary_hue="violet"),
        css=".gradio-container { max-width: 1100px; margin: auto; }",
    ) as app:
        gr.Markdown(
            "# SoloStudio-AI\n"
            "**One creator. Every character.**  "
            "Perform, and let the AI transfer your motion and voice onto your cast."
        )
        build_record_tab()
        build_gallery_tab()
        build_render_tab()
        build_outputs_tab()
    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
