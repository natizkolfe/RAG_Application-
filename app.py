"""NLEAD RAG chat UI with logo and theme choices."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from rag_chat import OLLAMA_MODEL, ask_rag, get_collection

PROJECT_DIR = Path(__file__).resolve().parent
LOGO_PATH = PROJECT_DIR / "assets" / "nlead_logo.png"

THEME_CHOICES = ("Navy", "Light", "Dark", "Slate")

CUSTOM_CSS = """
.theme-navy {
  --nlead-bg: #eef2f6;
  --nlead-panel: #ffffff;
  --nlead-ink: #0b1f3a;
  --nlead-muted: #5b6b7c;
  --nlead-accent: #c4a35a;
  --nlead-border: #d5dee8;
}

.theme-light {
  --nlead-bg: #f7f7f5;
  --nlead-panel: #ffffff;
  --nlead-ink: #1c2430;
  --nlead-muted: #667085;
  --nlead-accent: #1f6f5b;
  --nlead-border: #e4e7ec;
}

.theme-dark {
  --nlead-bg: #0f1419;
  --nlead-panel: #1a222d;
  --nlead-ink: #e8eef7;
  --nlead-muted: #9aa8b8;
  --nlead-accent: #d4b06a;
  --nlead-border: #2c3644;
}

.theme-slate {
  --nlead-bg: #e8edf2;
  --nlead-panel: #f5f7fa;
  --nlead-ink: #243447;
  --nlead-muted: #607083;
  --nlead-accent: #3d5a80;
  --nlead-border: #c9d4e0;
}

.gradio-container {
  max-width: 960px !important;
  margin: 0 auto !important;
}

#nlead-shell {
  background: var(--nlead-bg) !important;
  color: var(--nlead-ink) !important;
  min-height: 100vh;
  padding: 1.25rem 1rem 2rem;
  border-radius: 0;
}

#nlead-header {
  display: flex !important;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.9rem;
  padding: 1rem 1.1rem;
  background: var(--nlead-panel) !important;
  border: 1px solid var(--nlead-border) !important;
  border-radius: 16px;
}

#nlead-logo {
  width: 76px !important;
  min-width: 76px !important;
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid var(--nlead-border);
  background: #fff;
}

#nlead-logo img {
  width: 76px !important;
  height: 76px !important;
  object-fit: cover;
}

#nlead-title h1 {
  font-size: 1.55rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.02em;
  color: var(--nlead-ink) !important;
  margin: 0 !important;
}

#nlead-title p, #nlead-title span {
  margin: 0.3rem 0 0 !important;
  color: var(--nlead-muted) !important;
  font-size: 0.95rem !important;
}

.nlead-panel {
  background: var(--nlead-panel) !important;
  border: 1px solid var(--nlead-border) !important;
  border-radius: 16px !important;
  padding: 0.75rem !important;
  margin-bottom: 0.85rem;
}

#theme-row label {
  color: var(--nlead-ink) !important;
}

footer { display: none !important; }
"""


def respond(message: str, history: list) -> str:
    question = (message or "").strip()
    if not question:
        return "Please enter a question."
    try:
        return ask_rag(question, n_results=4)
    except Exception as exc:
        return f"Error: {exc}"


def set_theme(theme_name: str):
    key = (theme_name or "Navy").strip().lower()
    if key not in {"navy", "light", "dark", "slate"}:
        key = "navy"
    return gr.update(elem_classes=[f"theme-{key}"])


def main():
    print("Loading embedding model + Chroma...", flush=True)
    _, collection = get_collection()
    count = collection.count()
    print(f"Indexed chunks: {count}", flush=True)
    print(f"Ollama model: {OLLAMA_MODEL}", flush=True)

    theme = gr.themes.Soft(
        primary_hue="slate",
        secondary_hue="stone",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Source Sans 3"),
    ).set(
        body_background_fill="#eef2f6",
        block_background_fill="#ffffff",
        border_color_primary="#d5dee8",
        button_primary_background_fill="#0b1f3a",
        button_primary_background_fill_hover="#16375f",
        button_primary_text_color="#ffffff",
    )

    with gr.Blocks(title="NLEAD RAG Chat") as demo:
        shell = gr.Column(elem_id="nlead-shell", elem_classes=["theme-navy"])
        with shell:
            with gr.Row(elem_id="nlead-header"):
                if LOGO_PATH.exists():
                    gr.Image(
                        value=str(LOGO_PATH),
                        show_label=False,
                        container=False,
                        height=76,
                        width=76,
                        elem_id="nlead-logo",
                    )
                with gr.Column(elem_id="nlead-title"):
                    gr.Markdown("# NLEAD RAG Assistant")
                    gr.Markdown(
                        "Interactive knowledge chat for SharePoint lists and documents."
                    )

            with gr.Row(elem_classes=["nlead-panel"]):
                theme_picker = gr.Radio(
                    choices=list(THEME_CHOICES),
                    value="Navy",
                    label="Theme",
                    info="Choose a look for your demo",
                    elem_id="theme-row",
                )

            gr.ChatInterface(
                fn=respond,
                examples=[
                    "What feedback themes appear around leadership and conflict resolution?",
                    "Summarize key points from the CTC comprehensive report.",
                    "What courses or curriculum topics are available?",
                ],
                chatbot=gr.Chatbot(
                    label="NLEAD Chat",
                    height=480,
                    elem_classes=["nlead-panel"],
                ),
                textbox=gr.Textbox(
                    placeholder="Ask about AARs, curriculum, surveys, or documents...",
                    label="Message",
                ),
            )

            theme_picker.change(fn=set_theme, inputs=theme_picker, outputs=shell)

    print("Starting chat UI...", flush=True)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
        theme=theme,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
