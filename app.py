import gradio as gr
from encoder import BytePairEncoder
import random
import colorsys

class ColorTokenizer:
    def __init__(self, model_path: str):
        self.encoder = BytePairEncoder.load_from_file(model_path)
        self.color_cache = {}  # Cache colors for tokens
        
    def _get_token_color(self, token: str) -> str:
        """Generate and cache a consistent pastel color for a token"""
        if token not in self.color_cache:
            # Generate pastel color using HSV
            hue = random.random()  # Random hue
            saturation = 0.3  # Low saturation for pastel
            value = 1.0  # High value for brightness
            
            # Convert to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # Convert to hex
            self.color_cache[token] = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255), 
                int(rgb[1] * 255), 
                int(rgb[2] * 255)
            )
        return self.color_cache[token]
    
    def tokenize_with_colors(self, text: str) -> tuple[str, dict]:
        """Tokenize text and return colored HTML representation and stats"""
        if not text.strip():
            return "", {"Tokens": 0, "Characters": 0, "Compression": 0}
        
        # Tokenize
        tokens = self.encoder.tokenize(text)
        
        # Generate HTML with colored tokens
        html_parts = []
        for token in tokens:
            color = self._get_token_color(token)
            html_parts.append(
                f'<span style="background-color: {color}; '
                f'padding: 0.2em; margin: 0.1em; border-radius: 0.2em;">{token}</span>'
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
    html, stats = tokenizer.tokenize_with_colors(text)
    return html, stats

# Initialize tokenizer
tokenizer = ColorTokenizer("assets/hindi_tokenizer.json")

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
    Each token is highlighted with a different pastel color.
    The tokenizer has a vocabulary of 4,500 tokens and achieves ~3.5x compression on Hindi text.
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
