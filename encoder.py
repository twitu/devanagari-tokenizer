from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt
import json


class TrieNode:
    """Node in the prefix tree (trie) for fast token matching"""

    def __init__(self):
        self.children = {}
        self.is_token = False
        self.token = None


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
        self.stats["max_token_lengths"] = [1]  # Track evolution of max length

    def get_digram_stats(self) -> Counter:
        """Get digram counts"""
        counts = Counter()
        for pair in zip(self.data, self.data[1:]):
            pair = (int(pair[0]), int(pair[1]))
            counts[pair] += 1
        return counts

    def replace_byte_pair_in_data(self, pair: tuple[int, int], new_token: int) -> list:
        """Replace given byte pair with new token in data"""
        result = []
        i = 0
        while i < len(self.data):
            if (
                i < len(self.data) - 1
                and self.data[i] == pair[0]
                and self.data[i + 1] == pair[1]
            ):
                result.append(new_token)
                i += 2
            else:
                result.append(self.data[i])
                i += 1
        return result

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

    def merge_byte_pair(self) -> tuple[int, str, int]:
        """
        Merge the top byte pair

        Returns:
            tuple: (new_token_id, new_token_str, merge_count) or None if no more pairs to merge
        """
        # Get pair frequencies
        stats = self.get_digram_stats()
        if not stats:  # No more pairs to merge
            return None

        # Find most frequent pair
        (top_pair, count) = max(stats.items(), key=lambda x: x[1])

        # Add new token to vocabulary
        new_idx = self.encode_pair(top_pair)

        # Replace pairs in data
        self.data = self.replace_byte_pair_in_data(top_pair, new_idx)

        # Update statistics
        self.update_stats(count, self.itos[new_idx])

        return new_idx, self.itos[new_idx], count

    def print_progress(self, iteration: int, new_token: str, merge_count: int):
        """
        Print training progress in text format

        Args:
            iteration: Current iteration number
            new_token: Newly created token
            merge_count: Number of merges for this token
        """
        print(f"\nIteration {iteration:,}")
        print(f"Created token: '{new_token}' (merged {merge_count:,} times)")
        print(f"Current vocabulary size: {len(self.itos):,}")
        print(f"Current data size: {len(self.data):,}")
        print(f"Current compression ratio: {self.stats['compression_ratios'][-1]:.2f}")
        print("-" * 80)

    def encode_to_vocab_size(
        self,
        target_vocab_size: int,
        plot_interval: int = None,
        print_interval: int = 100,
    ) -> None:
        """
        Perform BPE encoding until reaching target vocabulary size

        Args:
            target_vocab_size: Maximum vocabulary size to reach
            plot_interval: How often to plot statistics (None for no plotting)
            print_interval: How often to print progress (None for no printing)
        """
        pbar = tqdm(
            total=target_vocab_size,
            desc=f"Encoding byte pairs",
            initial=len(self.chars),
            position=0,
            leave=True,
        )

        iteration = 0
        while len(self.itos) < target_vocab_size:
            # Train one iteration
            result = self.merge_byte_pair()
            if result is None:  # No more pairs to merge
                print("No more pairs to merge!")
                break

            new_idx, new_token, merge_count = result
            iteration += 1

            # Update progress bar
            pbar.update(1)

            # Print progress at intervals if requested
            if print_interval and iteration % print_interval == 0:
                self.print_progress(iteration, new_token, merge_count)

            # Plot statistics at intervals if requested
            if plot_interval and iteration % plot_interval == 0:
                self.plot_statistics(iteration=iteration, live_plot=True)

        pbar.close()

        # Final statistics
        print(f"\nTraining completed after {iteration:,} iterations")
        print(f"Final vocabulary size: {len(self.itos):,}")

    def plot_statistics(self, iteration: int = None, live_plot: bool = False):
        """
        Visualize the encoding statistics

        Args:
            iteration: Current iteration number (for live plotting)
            live_plot: Whether to clear output for live updates
        """
        if live_plot:
            from IPython.display import clear_output

            clear_output(wait=True)

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
        if self.stats["merge_counts"]:  # Only plot if we have merge counts
            ax3.hist(self.stats["merge_counts"], bins=30)
            ax3.set_xlabel("Number of Merges")
            ax3.set_ylabel("Frequency")
            ax3.set_title("Distribution of Merge Counts")

        # Plot 4: Token Lengths Over Time
        if self.stats["tokens_created"]:  # Only plot if we have tokens
            token_lengths = [len(token) for token in self.stats["tokens_created"]]
            ax4.plot(range(len(token_lengths)), token_lengths)
            ax4.set_xlabel("Merge Operation")
            ax4.set_ylabel("New Token Length")
            ax4.set_title("Token Length Evolution")

        plt.tight_layout()

        # Print current statistics if live plotting
        if iteration is not None:
            print(f"\nIteration {iteration}")
            print(f"Current vocabulary size: {len(self.itos):,}")
            print(f"Current data size: {len(self.data):,}")
            print(
                f"Current compression ratio: {self.stats['compression_ratios'][-1]:.2f}"
            )

        plt.show()

    def encode(self, text: str) -> list[int]:
        """Convert text to token indices"""
        return [self.stoi[c] for c in text]

    def decode(self, token_ids: list[int]) -> str:
        """Convert token indices back to text"""
        return "".join(self.itos[idx] for idx in token_ids)

    def save_to_file(self, filepath: str) -> None:
        """
        Save encoder state to a JSON file.

        Args:
            filepath: Path where to save the encoder state
        """
        state = {
            "chars": self.chars,
            "stoi": self.stoi,  # Only save stoi, we can reconstruct itos
            "max_token_length": self.max_token_length,
            "stats": self.stats,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        print(f"Encoder saved to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str) -> "BytePairEncoder":
        """
        Load encoder state from a JSON file.

        Args:
            filepath: Path to the saved encoder state

        Returns:
            BytePairEncoder: New instance with loaded state
        """
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)

        # Create a dummy instance (we'll override its state)
        instance = cls("")

        # Restore state
        instance.chars = state["chars"]
        instance.stoi = state["stoi"]
        instance.itos = {
            i: s for s, i in state["stoi"].items()
        }  # Reconstruct itos from stoi
        instance.max_token_length = state["max_token_length"]
        instance.stats = state["stats"]

        return instance

    def continue_training(
        self,
        text: str,
        target_vocab_size: int,
        plot_interval: int = None,
        print_interval: int = 100,
    ) -> None:
        """
        Continue training from current vocabulary state using new text

        Args:
            text: New text to train on
            target_vocab_size: New target vocabulary size
            plot_interval: How often to plot statistics
            print_interval: How often to print progress
        """
        # Store new original length for compression ratio
        self.original_length = len(text)

        # First encode the new text using current vocabulary
        print("Encoding new text with current vocabulary...")
        tokenizer = GreedyBPE(self)

        # Convert text to token IDs with progress bar
        self.data = [
            self.stoi[token]
            for token in tqdm(
                tokenizer.tokenize(text),
                total=len(text),
                desc="Tokenizing text",
                leave=False,
            )
        ]

        # Continue training
        print(
            f"\nContinuing training from {len(self.itos):,} to {target_vocab_size:,} tokens..."
        )
        pbar = tqdm(
            total=target_vocab_size,
            desc=f"Encoding byte pairs",
            initial=len(self.itos),
            position=0,
            leave=True,
        )

        iteration = 0
        while len(self.itos) < target_vocab_size:
            # Train one iteration
            result = self.merge_byte_pair()
            if result is None:  # No more pairs to merge
                print("No more pairs to merge!")
                break

            new_idx, new_token, merge_count = result
            iteration += 1

            # Update progress bar
            pbar.update(1)

            # Print progress at intervals if requested
            if print_interval and iteration % print_interval == 0:
                self.print_progress(iteration, new_token, merge_count)

            # Plot statistics at intervals if requested
            if plot_interval and iteration % plot_interval == 0:
                self.plot_statistics(iteration=iteration, live_plot=True)

        pbar.close()
        print(f"\nTraining completed. Final vocabulary size: {len(self.itos):,}")


def _tokenize_chunk(text: str, stoi: dict, max_token_length: int) -> list[str]:
    """Helper function for parallel tokenization that takes explicit arguments"""
    tokens = []
    while len(text) > 0:
        # Try prefixes of increasing length
        best_token = None
        prefix_length = min(len(text), max_token_length)

        for length in range(1, prefix_length + 1):
            prefix = text[:length]
            if prefix in stoi:
                best_token = prefix

        if best_token is None:
            tokens.append(text[0])
            text = text[1:]
        else:
            tokens.append(best_token)
            text = text[len(best_token) :]
    return tokens


class GreedyBPE:
    """Greedy tokenizer for Byte Pair Encoding"""

    def __init__(self, encoder: "BytePairEncoder", use_trie: bool = True):
        self.stoi = encoder.stoi
        self.max_token_length = encoder.max_token_length
        self.use_trie = use_trie
        self._trie_root = None

    def _build_trie(self) -> TrieNode:
        """Build a trie from the vocabulary for faster tokenization"""
        root = TrieNode()
        for token in self.stoi.keys():
            node = root
            for char in token:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_token = True
            node.token = token
        return root

    def tokenize(self, text: str) -> "TokenIterator":
        """
        Create an iterator over tokens in the text

        Args:
            text: Input text to tokenize
        Returns:
            Iterator yielding tokens
        """
        return TokenIterator(self, text)


class TokenIterator:
    """Iterator for greedy BPE tokenization"""

    def __init__(self, tokenizer: GreedyBPE, text: str):
        self.tokenizer = tokenizer
        self.text = text
        self.pos = 0

        # Build trie if needed and enabled
        if tokenizer.use_trie:
            if tokenizer._trie_root is None:
                tokenizer._trie_root = tokenizer._build_trie()
            self.use_trie = True
        else:
            self.use_trie = False

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self.pos >= len(self.text):
            raise StopIteration

        if self.use_trie:
            token = self._next_token_trie()
        else:
            token = self._next_token_simple()

        self.pos += len(token)
        return token

    def _next_token_simple(self) -> str:
        """Find next token using simple prefix matching"""
        best_token = None
        prefix_length = min(len(self.text) - self.pos, self.tokenizer.max_token_length)

        for length in range(1, prefix_length + 1):
            prefix = self.text[self.pos : self.pos + length]
            if prefix in self.tokenizer.stoi:
                best_token = prefix

        return best_token or self.text[self.pos]

    def _next_token_trie(self) -> str:
        """Find next token using trie matching"""
        node = self.tokenizer._trie_root
        longest_token = None

        for i in range(
            self.pos, min(len(self.text), self.pos + self.tokenizer.max_token_length)
        ):
            char = self.text[i]
            if char not in node.children:
                break
            node = node.children[char]
            if node.is_token:
                longest_token = node.token

        return longest_token or self.text[self.pos]


# # Example usage
# text <-- load text from dataset
# encoder = BytePairEncoder(text)
# encoder.encode_to_vocab_size(vocab_size)
# encoder.plot_statistics()
