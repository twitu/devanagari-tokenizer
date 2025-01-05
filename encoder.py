from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt


class BytePairEncoder:
    def __init__(self, text: str):
        # Initialize vocabulary from characters
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        # Initial encoding of text
        self.data = [self.stoi[c] for c in text]

        # Statistics tracking
        self.stats = {
            "vocab_sizes": [len(self.chars)],
            "data_sizes": [len(self.data)],
            "compression_ratios": [1.0],
            "merge_counts": [],
            "tokens_created": [],
            "max_token_lengths": [1],
        }

        # Store original length for compression ratio
        self.original_length = len(self.data)

        # Add max token length tracking
        self.max_token_length = 1  # Start with single characters
        self.stats['max_token_lengths'] = [1]  # Track evolution of max length

    def get_digram_stats(self):
        """Get digram counts"""
        counts = Counter()
        for pair in zip(self.data, self.data[1:]):
            # Convert tensor elements to integers and create a tuple
            pair = (int(pair[0]), int(pair[1]))
            counts[pair] += 1
        return counts

    def replace_byte_pair_in_data(
        self, pair: tuple[int, int], new_token: int
    ) -> list[int]:
        """Replace given byte pair with new token in data"""
        data_list = self.data
        i = 0
        while i < len(data_list) - 1:
            if data_list[i] == pair[0] and data_list[i + 1] == pair[1]:
                data_list[i] = new_token
                del data_list[i + 1]
            else:
                i += 1

        return data_list

    def encode_pair(self, pair: tuple[int, int]) -> int:
        """Add a new token to vocabulary from pair"""
        pair_str = self.itos[pair[0]] + self.itos[pair[1]]
        next_idx = len(self.itos)
        self.stoi[pair_str] = next_idx
        self.itos[next_idx] = pair_str
        
        # Update max token length
        self.max_token_length = max(self.max_token_length, len(pair_str))
        return next_idx

    def update_stats(self, merge_count: int, new_token: str):
        """Record statistics after each merge operation"""
        self.stats["vocab_sizes"].append(len(self.itos))
        self.stats["data_sizes"].append(len(self.data))
        self.stats["compression_ratios"].append(self.original_length / len(self.data))
        self.stats["merge_counts"].append(merge_count)
        self.stats["tokens_created"].append(new_token)

    def encode_to_vocab_size(self, target_vocab_size: int) -> None:
        """Perform BPE encoding until reaching target vocabulary size"""
        initial_vocab_size = len(self.itos)
        pbar = tqdm(
            total=target_vocab_size,
            desc="Encoding byte pairs",
            initial=len(self.chars),
            position=0,
            leave=True,
        )

        while len(self.itos) < target_vocab_size:
            # Get pair frequencies
            stats = self.get_digram_stats()
            if not stats:  # No more pairs to merge
                print("No more pairs to merge!")
                break

            # Find most frequent pair
            (top_pair, count) = max(stats.items(), key=lambda x: x[1])

            # Add new token to vocabulary
            new_idx = self.encode_pair(top_pair)

            # Replace pairs in data
            self.data = self.replace_byte_pair_in_data(top_pair, new_idx)

            # Update statistics
            self.update_stats(count, self.itos[new_idx])

            # Print progress
            # print(
            #     f"Encoded {count:,} occurrences of '{self.itos[top_pair[0]]}{self.itos[top_pair[1]]}'"
            # )

            # Update progress bar
            pbar.update(1)

        pbar.close()
        print(f"\nFinal vocabulary size: {len(self.itos):,}")

    def plot_statistics(self):
        """Visualize the encoding statistics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Vocabulary Size vs Data Size
        ax1.plot(self.stats["vocab_sizes"], self.stats["data_sizes"])
        ax1.set_xlabel("Vocabulary Size")
        ax1.set_ylabel("Dataset Size")
        ax1.set_title("Vocabulary Size vs Dataset Size")

        # Plot 2: Compression Ratio vs Vocabulary Size
        ax2.plot(self.stats["vocab_sizes"], self.stats["compression_ratios"])
        ax2.set_xlabel("Vocabulary Size")
        ax2.set_ylabel("Compression Ratio")
        ax2.set_title("Compression Ratio vs Vocabulary Size")

        # Plot 3: Merge Counts Distribution
        ax3.hist(self.stats["merge_counts"], bins=30)
        ax3.set_xlabel("Number of Merges")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Distribution of Merge Counts")

        # Plot 4: Token Lengths Over Time
        token_lengths = [len(token) for token in self.stats["tokens_created"]]
        ax4.plot(range(len(token_lengths)), token_lengths)
        ax4.set_xlabel("Merge Operation")
        ax4.set_ylabel("New Token Length")
        ax4.set_title("Token Length Evolution")

        # Add max token length evolution plot
        ax4.plot(self.stats['vocab_sizes'], self.stats['max_token_lengths'])
        ax4.set_xlabel('Vocabulary Size')
        ax4.set_ylabel('Maximum Token Length')
        ax4.set_title('Maximum Token Length Evolution')

        plt.tight_layout()
        plt.show()

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text by checking all possible prefixes up to max_token_length
        and selecting the longest matching token.
        
        Args:
            text: Input text to tokenize
        Returns:
            List of tokens
        """
        tokens = []
        while len(text) > 0:
            # Try prefixes of increasing length up to max_token_length
            best_token = None
            prefix_length = min(len(text), self.max_token_length)
            
            for length in range(1, prefix_length + 1):
                prefix = text[:length]
                if prefix in self.stoi:
                    best_token = prefix
            
            if best_token is None:
                # No token found - take first character
                tokens.append(text[0])
                text = text[1:]
            else:
                # Use the longest matching token found
                tokens.append(best_token)
                text = text[len(best_token):]
        
        return tokens

    def encode(self, text: str) -> list[int]:
        """Convert text to token indices"""
        return [self.stoi[token] for token in self.tokenize(text)]

    def decode(self, token_ids: list[int]) -> str:
        """Convert token indices back to text"""
        return "".join(self.itos[idx] for idx in token_ids)

# # Example usage
# text <-- load text from dataset
# encoder = BytePairEncoder(text)
# encoder.encode_to_vocab_size(vocab_size)
# encoder.plot_statistics()
