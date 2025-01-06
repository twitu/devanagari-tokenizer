import gradio as gr
from encoder import BytePairEncoder
from encoder import GreedyBPE

class TokenVisualizer:
    def __init__(self, model_path: str):
        encoder = BytePairEncoder.load_from_file(model_path)
        self.tokenizer = GreedyBPE(encoder)

    def tokenize_with_styling(self, text: str) -> tuple[str, dict]:
        """Tokenize text and return styled HTML representation and stats"""
        if not text.strip():
            return "", {"Tokens": 0, "Characters": 0, "Compression": 0}

        # Tokenize
        tokens = self.tokenizer(text)

        # Generate HTML with styled tokens
        html_parts = []
        for token in tokens:
            html_parts.append(
                f'<span style="'
                f'border: 1px solid #ccc; '
                f'border-radius: 4px; '
                f'padding: 0.2em 0.4em; '
                f'margin: 0.1em; '
                f'display: inline-block; '
                f'background-color: #f8f9fa; '
                f'">{token}</span>'
            )

        # Calculate statistics
        stats = {
            "Tokens": len(tokens),
            "Characters": len(text),
            "Compression": f"{len(text)/len(tokens):.2f}x"
        }

        return " ".join(html_parts), stats

def process_text(text: str) -> tuple[gr.HTML, dict]:
    """Process input text and return visualization and stats"""
    html, stats = tokenizer.tokenize_with_styling(text)
    return html, stats

# Initialize tokenizer
tokenizer = TokenVisualizer("assets/hindi_tokenizer_5100.json")

# Create Gradio interface
demo = gr.Interface(
    fn=process_text,
    inputs=[
        gr.Textbox(
            label="Input Hindi Text", 
            placeholder="यहाँ हिंदी में टेक्स्ट लिखें...",
            lines=3
        )
    ],
    outputs=[
        gr.HTML(label="Tokenized Output"),
        gr.JSON(label="Statistics")
    ],
    title="Hindi BPE Tokenizer Visualization",
    description="""
    This tokenizer breaks Hindi text into subword units using Byte Pair Encoding (BPE).
    Each token is displayed in a separate box.
    The tokenizer has a vocabulary of 5,100 tokens and achieves 3.7 compression on Hindi text.
    """,
    examples=[
        ["नमस्ते दुनिया"],
        ["मैं हिंदी सीख रहा हूं"],
        ["भारत एक महान देश है"],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch() 
