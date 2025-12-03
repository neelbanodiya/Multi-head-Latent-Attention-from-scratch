""""Model Configuration:
├─ Vocabulary: ~65 characters (a-z, A-Z, 0-9, punctuation)
├─ Context length: 256 tokens (small enough for Colab)
├─ Model dimension (d_model): 128
├─ Number of layers: 4
├─ Number of heads: 4
├─ Head dimension: 32 (d_model / num_heads)
├─ Latent dimension: 16 (compression: 64→16, 4x savings!)
├─ Batch size: 32
├─ Dataset: Tiny Shakespeare (~1MB, trains in minutes)
└─ Total parameters: ~2-3M (fits easily in Colab)"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from typing import Tuple, List, Dict

#Opening my text file that contain the shakespeare text
with open("data/shakespeare.txt", "r") as file:
  data = file.read()


class CharacterTokenizer:
    """
    Simple character-level tokenizer
    Maps each unique character to an integer ID
    """
    def __init__(self, text: str):
        # Get all unique characters and sort them
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        # Create character to index mapping
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {''.join(chars)}")

    def encode(self, text: str) -> List[int]:
        """Convert text to list of integers"""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices: List[int]) -> str:
        """Convert list of integers back to text"""
        return ''.join([self.idx_to_char[i] for i in indices])


class CharacterDataset(Dataset):

    def __init__(self,data: List[int], context_length:int):

        self.data = data
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self,idx):
        """Returns:
        x: input sequence [idx : idx+context_length]
        y: target sequence [idx+1 : idx+context_length+1]"""
        # Input: characters from idx to idx+context_length
        x = torch.tensor(self.data[idx : idx + self.context_length], dtype=torch.long)

        # Target: next character for each position (shifted by 1)
        y = torch.tensor(self.data[idx + 1 : idx + self.context_length + 1], dtype=torch.long)

        return x, y

def prepare_shakespeare_data(
    context_length: int = 256,
    train_split: float = 0.9,
    batch_size: int = 32,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, CharacterTokenizer, Dict]:
    """
    Complete pipeline to prepare Tiny Shakespeare data

    Args:
        context_length: Number of characters in each sequence
        train_split: Fraction of data to use for training (rest is validation)
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader (use 2 for Colab)

    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        tokenizer: CharacterTokenizer instance
        info: Dictionary with dataset statistics
    """

    # Create tokenizer
    print("\n2. Building character tokenizer...")
    tokenizer = CharacterTokenizer(data)

    # Encode the entire text
    print("\n3. Encoding text...")
    encoded_data = tokenizer.encode(data)
    print(f"   ✓ Encoded length: {len(encoded_data):,}")

    # Split into train and validation
    print("\n4. Splitting data...")
    split_idx = int(len(encoded_data) * train_split)
    train_data = encoded_data[:split_idx]
    val_data = encoded_data[split_idx:]

    print(f"   ✓ Train size: {len(train_data):,} characters")
    print(f"   ✓ Val size: {len(val_data):,} characters")
    print(f"   ✓ Train sequences: {len(train_data) - context_length:,}")
    print(f"   ✓ Val sequences: {len(val_data) - context_length:,}")


    print("\n5. Creating PyTorch datasets...")
    train_dataset = CharacterDataset(train_data, context_length)
    val_dataset = CharacterDataset(val_data, context_length)

    print("\n6. Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Gather statistics
    info = {
        'total_chars': len(data),
        'vocab_size': tokenizer.vocab_size,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'context_length': context_length,
        'batch_size': batch_size,
        'train_batches': len(train_loader),
        'val_batches': len(val_loader)
    }

    return train_loader, val_loader, tokenizer, info

def test_dataloader(train_loader: DataLoader, tokenizer: CharacterTokenizer):
    """
    Test function to verify the dataloader works correctly
    Shows a sample batch and decodes it
    """
    print("\n" + "=" * 60)
    print("TESTING DATALOADER")
    print("=" * 60)

    # Get one batch
    batch_x, batch_y = next(iter(train_loader))

    print(f"\nBatch shapes:")
    print(f"   Input (x): {batch_x.shape}  [batch_size, context_length]")
    print(f"   Target (y): {batch_y.shape}  [batch_size, context_length]")

    # Show first sequence
    print(f"\nFirst sequence in batch:")
    print(f"   Input IDs: {batch_x[0][:20].tolist()}...")
    print(f"   Target IDs: {batch_y[0][:20].tolist()}...")

    # Decode and display
    input_text = tokenizer.decode(batch_x[0].tolist())
    target_text = tokenizer.decode(batch_y[0].tolist())

    print(f"\nDecoded (first 100 chars):")
    print(f"   Input:  '{input_text[:100]}'")
    print(f"   Target: '{target_text[:100]}'")

    # Verify target is input shifted by 1
    print(f"\nVerification (target should be input shifted by 1):")
    print(f"   Input char 0: '{input_text[0]}' -> Target char 0: '{target_text[0]}'")
    print(f"   Input char 1: '{input_text[1]}' -> Target char 1: '{target_text[1]}'")
    print(f"   Match: {input_text[1:] == target_text[:-1]}")

    print("\n" + "=" * 60)


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'context_length': 256,    # Sequence length
        'train_split': 0.9,       # 90% train, 10% validation
        'batch_size': 32,         # Good for Colab T4 GPU
        'num_workers': 2          # For data loading
    }

    # Prepare data
    train_loader, val_loader, tokenizer, info = prepare_shakespeare_data(**CONFIG)

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    for key, value in info.items():
        print(f"{key:20s}: {value:,}")

    # Test the dataloader
    test_dataloader(train_loader, tokenizer)

    print("\n✅ Data preparation successful!")
    print("✅ Ready to train the model!")