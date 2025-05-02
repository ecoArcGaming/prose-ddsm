from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from PoET.poet.alphabets import Alphabet, Uniprot21
import random

from collections import defaultdict

IGNORE_INDEX = -100


class ATCG(Alphabet):
    """
    tokenize DNA sequence
    """

    def __init__(
        self,
        mask=True,
        include_gap=True,
        include_startstop=True,
        distinct_startstop=True,
    ):
        chars = b"ATCG"
        gap_token = start_token = stop_token = -1
        if include_gap:
            chars = chars + b"-"
            gap_token = len(chars) - 1
        if include_startstop:
            chars = chars + b"*"
            start_token = stop_token = len(chars) - 1
        if distinct_startstop:
            chars = chars + b"$"
            stop_token = len(chars) - 1
        mask_token = len(chars)
        encoding = np.arange(len(chars))
        missing = mask_token

        super(ATCG, self).__init__(chars, encoding=encoding, mask=mask, missing=missing)

        self.gap_token = gap_token
        self.start_token = start_token
        self.stop_token = stop_token
        self.mask_token = mask_token


class PromoterDataset(Dataset):
    """
    Dataloader for promoter sets from Zoonomia
    """

    def __init__(self, sequences: dict, queries: dict, alphabet, max_length):
        self.alphabet = alphabet
        self.sequences = {k: list(v) for k, v in sequences.items()}
        self.queries = queries
        self.num_special_characters = 2
        self.ids = list(sequences.keys())
        self.sampling_weights = [len(v) for k, v in self.sequences.items()]
        total = sum(self.sampling_weights)
        self.sampling_weights = [i / total for i in self.sampling_weights]
        self.max_len = max_length
        self.iters = len(self.ids) * 5

        self.counter = defaultdict(int)

    def __getitem__(self, idx):
        """
        randomly sample a gene with replacement
        weighted by the number of sequences the gene has
        """
        id = np.random.choice(self.ids, p=self.sampling_weights)
        self.counter[id] += 1

        sampled_set = self.sample_and_fit_sequences(
            id,
            max_seq_length=self.max_len,
        )
        
        # sampled_set = self.diverse_sample_and_fit_sequences(id, max_seq_length=self.max_len)
        return self.pack_inputs(sampled_set, self.alphabet, self.max_len)

    def get_inference_seqs(self, variant: str, id: str):
        try:
            sampled_set = self.sample_and_fit_sequences(
                id, sequence=variant, max_seq_length=self.max_len
            )

            # sampled_set = self.diverse_sample_and_fit_sequences(id, max_seq_length=self.max_len, sequence=variant)
        except KeyError:
            return []
        return sampled_set["passages"]

    def sample_query(self, id: str, p_human: float) -> str:
        """
        sample a query sequence for a given gene, with a chosen probability of
        sampling a human sequence

        Args:
        id: gene name
        p_human: percent probability of sampling human

        Returns:
        single sampled DNA sequence as a query
        """
        human = np.random.choice([True, False], p=[p_human, 1 - p_human])
        if human:
            return self.queries[id]
        else:
            return np.random.choice(self.sequences[id])


    def sample_and_fit_sequences(
        self,
        query_id_or_sequence: Optional[
            str
        ] = None,  # Changed parameter name to be more explicit
        weights: Optional[List[float]] = None,
        max_seq_length: int = 1024,
        max_individual_seq_length: Optional[int] = None,
        include_query: bool = True,
        sample: bool = True,
        truncate: bool = False,
        chunk_size: int = 2,
        max_skips: int = 2,
        sequence=None,
    ) -> Dict[str, Any]:
        """
        Packs sequences until max length is reached. Can exclude query and use deterministic ordering.

        Args:
            query_id_or_sequence: Either a UniRef ID to lookup in fasta_dataset, or the actual sequence
            member_ids: List of UniRef IDs or sequences for passages
            weights: Optional sampling weights for passages
            fasta_dataset: Optional map from UniRef IDs to sequences. If None, assumes inputs are sequences
            max_seq_length: Maximum length of all sequences combined
            max_individual_seq_length: Optional int to limit the length of individual sequences.
            include_query: Whether to include and process query sequence
            sample: Whether to sample randomly (True) or use sequential order (False)
            truncate: Whether to truncate the last sequence if it doesn't fit
            chunk_size: Number of IDs to process at a time for sampling
            max_skips: Max chunks allowed to add no passages before giving up

        Returns:
            Dict with id, sequence, passages, passage_lengths, query_length, and total_tokens
        """

        def truncate_sequence(seq: str, max_len: int) -> str:
            """Helper to truncate a sequence to max length while accounting for special tokens"""
            if len(seq) + self.num_special_characters > max_len:
                return seq[: max_len - self.num_special_characters]
            return seq

        if sequence is not None:
            query_sequence = sequence
        else:
            query_sequence = self.sample_query(
                query_id_or_sequence, 0.3
            )  # P(human sequence) = 0.3

        # Apply individual sequence length limit if specified
        if max_individual_seq_length and query_sequence:
            query_sequence = truncate_sequence(
                query_sequence, max_individual_seq_length
            )

        query_length = (
            len(query_sequence) + self.num_special_characters
            if query_sequence
            else None
        )

        # Calculate effective length for passages
        effective_length = max_seq_length - (
            query_length if query_length is not None else 0
        )

        passages = []
        passage_lengths = []
        total_tokens = query_length if query_length is not None else 0
        leftover = effective_length

        member_ids = self.sequences[query_id_or_sequence]

        if sample:
            member_weights = np.array(
                weights
                if weights is not None
                else [1.0 / len(member_ids)] * len(member_ids)
            )
            member_weights /= member_weights.sum()

            skip_rounds = 0
            while leftover > 0 and skip_rounds < max_skips:
                sampled_ids = np.random.choice(
                    member_ids, size=chunk_size, replace=True, p=member_weights
                )
                added = False

                for seq in sampled_ids:
                    if not seq:
                        continue

                    # Apply individual sequence length limit if specified
                    if max_individual_seq_length:
                        seq = truncate_sequence(seq, max_individual_seq_length)

                    seq_len = len(seq) + self.num_special_characters
                    if seq_len <= leftover:
                        passages.append(seq)
                        passage_lengths.append(seq_len)
                        leftover -= seq_len
                        total_tokens += seq_len
                        added = True
                    elif truncate and leftover > 0:
                        # Truncate to fit remaining space
                        trunc_len = leftover - self.num_special_characters
                        seq = seq[:trunc_len]
                        passages.append(seq)
                        passage_lengths.append(leftover)
                        total_tokens += leftover
                        leftover = 0
                        added = True
                        break

                    if leftover <= 0:
                        break

                skip_rounds = 0 if added else skip_rounds + 1

        return {
            "id": query_id_or_sequence if include_query else None,
            "sequence": query_sequence,
            "passages": passages,
            "passage_lengths": passage_lengths,
            "query_length": query_length,
            "total_tokens": total_tokens,
        }
    
    def greedy_sample_and_fit_sequences(
        self,
        query_id_or_sequence: Optional[
            str
        ] = None,  # Changed parameter name to be more explicit
        max_seq_length: int = 12048,
        max_individual_seq_length: Optional[int] = None,
        include_query: bool = True,
        sequence : str = None,
        p_human: float = 0.3
    ) -> Dict[str, Any]:
        """
        Iteratively samples a subset of a given total size by selecting sequences that
        maximize the average Hamming distance to the already selected sequences.

        Args:
            query_id_or_sequence: Either a UniRef ID to lookup in fasta_dataset, or the actual sequence
            max_seq_length: Maximum length of all sequences combined
            max_individual_seq_length: Optional int to limit the length of individual sequences.
            sequence: Optional provided query sequence 
            p_human: Optional probability of sampling a human query, if not given
        Returns:
            Dict with id, sequence, passages, passage_lengths, query_length, and total_tokens
        """

        def truncate_sequence(seq: str, max_len: int) -> str:
            """Helper to truncate a sequence to max length while accounting for special tokens"""
            if len(seq) + self.num_special_characters > max_len:
                return seq[: max_len - self.num_special_characters]
            return seq
        
        if sequence is not None:
            query_sequence = sequence
        else:
            query_sequence = self.sample_query(
                query_id_or_sequence, p_human)  # P(human sequence) = 0.3

        # Apply individual sequence length limit if specified
        if max_individual_seq_length and query_sequence:
            query_sequence = truncate_sequence(
                query_sequence, max_individual_seq_length)

        query_length = (len(query_sequence) + self.num_special_characters if query_sequence else None)

        # Calculate effective length for passages
        effective_length = max_seq_length - (
            query_length if query_length is not None else 0)

        passages = []
        passage_lengths = []
        total_tokens = query_length if query_length is not None else 0
        leftover = effective_length  # TOTAL Lengths of the homologs

        member_ids = self.sequences[query_id_or_sequence]
        n = len(member_ids)
        m = max(len(s) for s in member_ids)

        data = np.zeros((len(member_ids), m), dtype=np.uint32) # padded sequence ordinals
        for i, s in enumerate(member_ids):
            data[i, :len(s)] = [ord(c) for c in s]
        
        selected_indices = []
        available_mask = np.ones(n, dtype=bool)

        # Stores the *sum* of Hamming distances from each string to the selected set
        sum_distances_to_selected = np.zeros(n, dtype=np.float64)

        # Start with a random sequence
        start_index = random.randint(0, n - 1)
        seq_len = (len(member_ids[start_index]) + self.num_special_characters)
        leftover -= seq_len
        total_tokens += seq_len
        passage_lengths.append(seq_len)
        selected_indices.append(start_index)
        available_mask[start_index] = False
        num_selected = 1

        # Calculate initial distances from the first selected string to all others
        first_selected_data = data[start_index]
        # Get indices that are still available
        current_available_indices = np.where(available_mask)[0]
        if len(current_available_indices) > 0:
            
            # Calculate distances from available strings to the first selected one with broadcasting
            initial_distances = np.sum(data[current_available_indices] != first_selected_data, axis=1)
            # Update 
            sum_distances_to_selected[current_available_indices] = initial_distances
        
        # iterative greedy sampling
        while leftover > 0: 
            if not np.any(available_mask):
                break
            current_available_indices = np.where(available_mask)[0]

            # Calculate the *average* Hamming distance for each available candidate
            avg_distances_candidates = sum_distances_to_selected[current_available_indices] / num_selected
            best_candidate_local_idx = np.argmax(avg_distances_candidates)
            best_original_idx = current_available_indices[best_candidate_local_idx]

            # Add the best candidate to the selected set
            selected_indices.append(best_original_idx)
            available_mask[best_original_idx] = False

            num_selected += 1
            seq_len = (self.num_special_characters + len(member_ids[best_original_idx]))
            leftover -= seq_len
            total_tokens += seq_len
            passage_lengths.append(seq_len)
            # Update the sum_distances for the remaining available strings
            newly_selected_data = data[best_original_idx]
            remaining_available_indices = np.where(available_mask)[0] # Indices still available *after* selecting the latest

            if len(remaining_available_indices) == 0:
                break
            # Calculate distances of the newly selected one
            distances_to_new = np.sum(data[remaining_available_indices] != newly_selected_data, axis=1)
            sum_distances_to_selected[remaining_available_indices] += distances_to_new

        passages = [member_ids[i] for i in selected_indices]

        return {
            "id": query_id_or_sequence if include_query else None,
            "sequence": query_sequence,
            "passages": passages,
            "passage_lengths": passage_lengths,
            "query_length": query_length,
            "total_tokens": total_tokens,
        }

    def pack_inputs(
        self,
        sample,
        tokenizer,  # DNA tokenizer instance
        max_seq_length,
        padding_idx=IGNORE_INDEX,
        reverse_sequence=False,
        skip_padding=True,
    ):
        """
        Packs a single sample's query and passages into a single sequence, using Uniprot21 tokenization.
        Each sequence is wrapped with start/stop tokens before concatenation.

        Args:
            sample (Dict[str, Any]): A dictionary containing:
                - "id": Query ID
                - "sequence": Query sequence string
                - "passages": List of passage sequence strings
                - "passage_lengths": List of passage lengths (including special tokens)
                - "query_length": Length of query sequence (including special tokens)
            tokenizer: Uniprot21 tokenizer instance
            max_seq_length (int): Maximum length of the packed sequence
            padding_idx (int): Token ID to use for padding
            cls_token (str): Start token character ($ for Uniprot21)
            eos_token (str): Stop token character (* for Uniprot21)
            reverse_sequence (bool): If True, reverse sequences before tokenization
            skip_padding (bool): If True, don't pad sequence to max_seq_length

        Returns:
            Dict[str, torch.Tensor]: Packed inputs including:
                - "tokens": (seq_len,) LongTensor of token IDs
                - "seq_lens": List[int] containing length of each sequence chunk
                - "labels": (seq_len,) LongTensor same as tokens but with padding masked to -100

        Example:
            For sequence ABC, passages [DEF, GH] and max_seq_length=14:

            Normal order (reverse_sequence=False):
                tokens:  [$ D E F * $ G H * $ A B C * <pad>]
                seq_lens: [5, 4, 5]  # Each sequence length includes its start/stop tokens
                labels:  [$ D E F * $ G H * $ A B C * -100]

            Reversed (reverse_sequence=True):
                tokens:  [* F E D $ * H G $ * C B A $ <pad>]
                seq_lens: [5, 4, 5]  # Lengths stay same, just content is reversed
                labels:  [* F E D $ * H G $ * C B A $ -100]

            Note: $ = start token, * = stop token, <pad> = padding token
                Actual tokens will be Uniprot21 token IDs, shown here as characters for clarity
                Sequences are concatenated: passages first, then query
        """

        def tokenize_sequence(text: str, reverse_sequence: bool) -> np.ndarray:
            if reverse_sequence:
                text = text[::-1]
            if isinstance(text, str):
                text_bytes = text.encode("ascii")
            else:
                text_bytes = text  # Already bytes
            tokens = tokenizer.encode(text_bytes)
            if reverse_sequence:
                return np.concatenate(
                    [[tokenizer.stop_token], tokens, [tokenizer.start_token]]
                )
            else:
                return np.concatenate(
                    [[tokenizer.start_token], tokens, [tokenizer.stop_token]]
                )

        # Initialize sequences and lengths
        all_sequences = []
        all_lengths = sample[
            "passage_lengths"
        ].copy()  # Use pre-calculated passage lengths

        # Process passages
        if sample["passages"]:
            for seq in sample["passages"]:
                if seq is not None:
                    tokens = tokenize_sequence(seq, reverse_sequence)
                    all_sequences.append(tokens)

        # Process query if it exists
        query_sequence = sample["sequence"]
        if query_sequence is not None:
            tokens = tokenize_sequence(query_sequence, reverse_sequence)
            all_sequences.append(tokens)
            # Use pre-calculated query length
            all_lengths.append(sample["query_length"])

        # Concatenate all sequences
        if all_sequences:
            tokens = np.concatenate(all_sequences)
            tokens = torch.from_numpy(tokens).long()
        else:
            tokens = torch.empty(0, dtype=torch.long)

        # Add padding if needed
        current_length = sum(all_lengths)
        if not skip_padding and current_length < max_seq_length:
            padding_length = max_seq_length - current_length
            all_lengths.append(padding_length)
            padding = torch.full((padding_length,), padding_idx, dtype=tokens.dtype)
            tokens = torch.cat([tokens, padding])

        # Create labels (same as tokens but with padding masked)
        return {
            "tokens": tokens,
            "seq_lens": all_lengths,
            "labels": tokens.clone()
        }

    def padded_collate_packed(
        self,
        batch: List[Dict[str, Any]],
        pad_mode: Optional[str] = None,
        fixed_length: Optional[int] = None,
        padding_idx: int = IGNORE_INDEX,
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of packed samples into a single dictionary.

        Args:
            batch: List of dictionaries containing packed sequences.
            fabric: Fabric instance for moving data to device.
            pad_mode: Can be "largest", "fixed", or None. If None, behavior is the default ("largest").
            fixed_length: If pad_mode=="fixed", pad each sample to this length.
            padding_idx: The token ID to use for padding.

        Returns:
            Dict with:
                - "tokens": (bsz, seq_len) LongTensor (padded)
                - "labels": (bsz, seq_len) LongTensor (padded)
                - "segment_sizes": (bsz, max_segments) LongTensor (padded with zeros)
        """
        # If pad_mode is None, default to "largest"
        if pad_mode is None:
            pad_mode = "largest"

        # Determine target length for tokens
        token_lengths = [x["tokens"].size(0) for x in batch]
        if pad_mode == "largest":
            target_length = max(token_lengths)
        elif pad_mode == "fixed":
            if fixed_length is None:
                raise ValueError(
                    "fixed_length must be provided when pad_mode is 'fixed'"
                )
            target_length = fixed_length
        else:
            raise ValueError("Unknown pad_mode. Use 'largest', 'fixed', or None.")

        padded_tokens = []
        padded_labels = []

        for x in batch:
            token = x["tokens"]
            label = x["labels"]
            pad_len = target_length - token.size(0)
            if pad_len > 0:
                token = F.pad(token, (0, pad_len), value=padding_idx)
                label = F.pad(label, (0, pad_len), value=padding_idx)
            padded_labels.append(label)
            padded_tokens.append(token)
        tokens_tensor = torch.stack(padded_tokens, dim=0)
        labels_tensor = torch.stack(padded_labels, dim=0)

        # Collate segment sizes (each sample may have several segments).
        max_segments = max(len(x["seq_lens"]) for x in batch)
        bsz = len(batch)
        segment_sizes = torch.zeros((bsz, max_segments), dtype=torch.int32)
        for i, x in enumerate(batch):
            seq_lens = torch.tensor(x["seq_lens"], dtype=torch.int32)
            segment_sizes[i, : len(seq_lens)] = seq_lens

        collated_batch = {
            "tokens": tokens_tensor,
            "segment_sizes": segment_sizes,
            "labels": labels_tensor,
        }

        collated_batch = self._to_cuda(collated_batch)
        return collated_batch

    def __len__(self):
        return self.iters

    def _to_cuda(self, tok_dict):
        return {k: v.long().cuda() for k, v in tok_dict.items()}

    @staticmethod
    def train_validation_split(
        chr: int, sequences: dict, queries: dict
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        return sequences, queries, sequences, queries

    @staticmethod
    def _train_validation_split(
        chr: int, sequences: dict, queries: dict
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        splits two dictionaries based on the chromosome number of the keys

        """
        train_seq = {}
        train_query = {}
        val_seq = {}
        val_query = {}
        chr_pattern = f"_chr{chr}"
        for key in sequences.keys():
            if key not in queries:
                continue
            if chr_pattern in key:
                val_seq[key] = sequences[key]
                val_query[key] = queries[key]
            else:
                train_seq[key] = sequences[key]
                train_query[key] = queries[key]

        print(f"training with {len(train_query)} sequences")
        print(f"validating with {len(val_query)} sequences")

        return train_seq, train_query, val_seq, val_query
