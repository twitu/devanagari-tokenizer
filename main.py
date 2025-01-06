# %%
import re
from pathlib import Path
from encoder import BytePairEncoder
from tqdm import tqdm
import emoji

# %%
def clean_text(text: str) -> str:
    """
    Clean text by:
    - Removing emojis and special characters
    - Keeping Hindi characters, numbers, and basic punctuation
    - Removing empty lines
    - Normalizing whitespace
    
    Args:
        text: Input text containing Hindi text with emojis and other characters
    Returns:
        Cleaned text with only Hindi characters and basic punctuation
    """
    # Remove emojis
    text = emoji.replace_emoji(text, '')
    
    # Keep only Hindi characters (including Hindi numbers) and basic punctuation
    # Unicode ranges:
    # \u0900-\u097F : Devanagari (Hindi)
    # \u0966-\u096F : Devanagari numbers
    # Basic punctuation and spaces
    text = re.sub(r'[^\u0900-\u097F\u0966-\u096F\s.,!?-]', '', text)
    
    # Remove empty lines and normalize whitespace
    lines = [line.strip() for line in text.split('\n')]
    lines = [re.sub(r'\s+', ' ', line) for line in lines]
    lines = [line for line in lines if line]  # Remove empty lines
    
    # Join with spaces and strip any leading/trailing whitespace
    return ' '.join(lines).strip()

def load_hindi_text(directory: str, max_files: int = None) -> str:
    """
    Load and clean Hindi text from files in directory
    
    Args:
        directory: Path to directory containing text files
        max_files: Maximum number of files to load (None for all files)
    """
    all_text = []
    
    # Convert to Path object for easier handling
    dir_path = Path(directory)
    
    # Get list of files and limit if specified
    files = list(dir_path.glob('*.txt'))
    if max_files:
        files = files[:max_files]
    
    # Process each file
    for file_path in tqdm(files, desc="Loading files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                cleaned_text = clean_text(text)
                if cleaned_text:  # Only add non-empty texts
                    all_text.append(cleaned_text)
        except UnicodeDecodeError:
            print(f"Warning: Skipping file {file_path} due to encoding issues")
    
    # Join all texts with space
    combined_text = ' '.join(all_text)
    
    print(f"Loaded {len(all_text):,} files")
    print(f"Total text length: {len(combined_text):,} characters")
    
    return combined_text

# %%
def main():
    # Configuration
    data_dir = "train/train"
    vocab_size = 4500
    output_file = "hindi_tokenizer.json"
    
    # Load and clean text
    print("Loading and cleaning text...")
    text = load_hindi_text(data_dir, 10000)
    
    # Create and train encoder
    print("\nTraining BPE encoder...")
    encoder = BytePairEncoder(text)
    encoder.encode_to_vocab_size(vocab_size, plot_interval=10, print_interval=100)
    
    # Save the encoder
    print("\nSaving encoder...")
    encoder.save_to_file(output_file)
    
    # Plot statistics
    print("\nPlotting statistics...")
    encoder.plot_statistics()
    
    # Print some example tokenizations
    print("\nExample tokenizations:")
    test_texts = [
        "नमस्ते दुनिया",
        "मैं हिंदी सीख रहा हूं",
        "भारत एक महान देश है"
    ]
    
    for text in test_texts:
        tokens = encoder.tokenize(text)
        print(f"\nText: {text}")
        print(f"Tokens: {tokens}")

# %%
def continue_training():
    # Configuration
    data_dir = "train/train"
    initial_model = "assets/hindi_tokenizer.json"
    final_model = "assets/hindi_tokenizer_5100.json"
    target_vocab_size = 5100
    
    # Load and clean more text
    print("Loading and cleaning additional text...")
    text = load_hindi_text(data_dir, max_files=10000)
    
    # Load existing encoder
    print("\nLoading existing encoder...")
    encoder = BytePairEncoder.load_from_file(initial_model)
    print(f"Current vocabulary size: {len(encoder.itos):,}")
    

    # Continue training
    print("\nContinuing training...")
    encoder.continue_training(
        text,
        target_vocab_size,
        plot_interval=20,
        print_interval=None,
    )
    
    # Save the updated encoder
    print("\nSaving encoder...")
    encoder.save_to_file(final_model)
    
    # Plot final statistics
    print("\nPlotting final statistics...")
    encoder.plot_statistics()
    
    # Print some example tokenizations
    print("\nExample tokenizations:")
    test_texts = [
        "नमस्ते दुनिया",
        "मैं हिंदी सीख रहा हूं",
        "भारत एक महान देश है"
    ]
    
    for text in test_texts:
        tokens = encoder.tokenize(text)
        print(f"\nText: {text}")
        print(f"Tokens: {tokens}")


# %%
continue_training()

# %%
from encoder import GreedyBPE

encoder = BytePairEncoder.load_from_file("assets/hindi_tokenizer_5100.json")
tokenizer = GreedyBPE(encoder)

print("\nExample tokenizations:")
test_texts = [
    "नमस्ते दुनिया",
    "मैं हिंदी सीख रहा हूं",
    "भारत एक महान देश है"
]
    
for text in test_texts:
    tokens = tokenizer.tokenize(text)
    print(f"\nText: {text}")
    print(f"Tokens: {' '.join(tokens)}")


# %%
