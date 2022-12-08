# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
import torch
print(torch.cuda.device_count())
from datasets import load_dataset
from dataclasses import dataclass
import pandas as pd
import argparse
import log
import math
from transformers import Trainer, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
import spacy
nlp = spacy.load("en_core_web_sm")

# from transformers.data import _torch_collate_batch
logger = log.get_logger('root')

@dataclass
class DynamicDataCollatorForLanguageModeling:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        # ori_text = self.tokenizer.decode(batch['input_ids'])
        tmp = batch['input_ids'] == batch['labels']
        batch['labels'][tmp] = -100
        return batch

# def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
#     """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
#     import numpy as np
#     import torch

#     # Tensorize if necessary.
#     if isinstance(examples[0], (list, tuple, np.ndarray)):
#         examples = [torch.tensor(e, dtype=torch.long) for e in examples]

#     length_of_first = examples[0].size(0)

#     # Check if padding is necessary.

#     are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
#     if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
#         return torch.stack(examples, dim=0)

#     # If yes, check if we have a `pad_token`.
#     if tokenizer._pad_token is None:
#         raise ValueError(
#             "You are attempting to pad samples but the tokenizer you are using"
#             f" ({tokenizer.__class__.__name__}) does not have a pad token."
#         )

#     # Creating the full tensor and filling it with our data.
#     max_length = max(x.size(0) for x in examples)
#     if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
#         max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
#     result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
#     for i, example in enumerate(examples):
#         if tokenizer.padding_side == "right":
#             result[i, : example.shape[0]] = example
#         else:
#             result[i, -example.shape[0] :] = example
#     return result

# @dataclass
# class DynamicDataCollatorForLanguageModeling:
#     tokenizer: PreTrainedTokenizerBase
#     mlm: bool = True
#     mlm_probability: float = 0.15
#     pad_to_multiple_of: Optional[int] = None
#     tf_experimental_compile: bool = False
#     return_tensors: str = "pt"
    
#     def __call__(self, features, return_tensors=None):
#         if return_tensors is None:
#             return_tensors = self.return_tensors
#         if return_tensors == "tf":
#             return self.tf_call(features)
#         elif return_tensors == "pt":
#             return self.torch_call(features)
#         elif return_tensors == "np":
#             return self.numpy_call(features)
#         else:
#             raise ValueError(f"Framework '{return_tensors}' not recognized!")

#     def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
#         # Handle dict or lists with proper padding and conversion to tensor.
#         if isinstance(examples[0], Mapping):
#             batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
#         else:
#             batch = {
#                 "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
#             }

#         # If special token mask has been preprocessed, pop it from the dict.
#         special_tokens_mask = batch.pop("special_tokens_mask", None)
#         if self.mlm:
#             batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
#                 batch["input_ids"], special_tokens_mask=special_tokens_mask
#             )
#         else:
#             labels = batch["input_ids"].clone()
#             if self.tokenizer.pad_token_id is not None:
#                 labels[labels == self.tokenizer.pad_token_id] = -100
#             batch["labels"] = labels
#         return batch

#     def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
#         """
#         Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
#         """
#         import torch

#         labels = inputs.clone()
#         # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
#         probability_matrix = torch.full(labels.shape, self.mlm_probability)
#         if special_tokens_mask is None:
#             special_tokens_mask = [
#                 self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
#             ]
#             special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
#         else:
#             special_tokens_mask = special_tokens_mask.bool()

#         probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
#         masked_indices = torch.bernoulli(probability_matrix).bool()
#         labels[~masked_indices] = -100  # We only compute loss on masked tokens

#         # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
#         indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
#         inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

#         # 10% of the time, we replace masked input tokens with random word
#         indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
#         random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
#         inputs[indices_random] = random_words[indices_random]

#         # The rest of the time (10% of the time) we keep the masked input tokens unchanged
#         return inputs, labels



def main():
    parser = argparse.ArgumentParser(
        description="Command line interface for ESG")

    # Required parameters
    parser.add_argument("--mask_stratagy", default=None, type=str, required=True,
                        help="random or dynamic")
    parser.add_argument("--data_path", default=None, type=str, required=True,
                        help="The input data path. Should contain the data files for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--batch_size", default=None, type=int, required=True,
                        help="Batch_size per gpu")
    parser.add_argument("--chunk_size", default=None, type=int, required=True,
                        help="The size of input to model")
    parser.add_argument("--training_size", default=None, type=int, required=True,
                        help="The number of example to be trained")                  
    parser.add_argument('--do_train', action='store_true',
                        help="Whether to perform training")
    parser.add_argument('--do_eval', action='store_true',
                        help="Whether to perform evaluation")
    parser.add_argument('--load_cache_dir', default=None, type=str, required=True,
                        help="Whether to load cache")
    parser.add_argument('--tokenizer_path', default=None, type=str, required=True,
                        help="tokenizer_path")
    
    args = parser.parse_args()
    logger.info("Parameters: {}".format(args))

    from transformers import AutoModelForMaskedLM
    from datasets import load_from_disk
    model_checkpoint = args.model_name_or_path
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    if not args.load_cache_dir:
        esg_dataset = load_dataset("csv", data_files=args.data_path)
        # esg_dataset  = esg_dataset.train_test_split(test_size=0.2, shuffle=True)
        print(esg_dataset)

        print("start to tokenize")
        def tokenize_function(examples):
            result = tokenizer(examples["Abstract"])
            if args.mask_stratagy == 'dynamic':
                label = tokenizer(examples["Label"])            
                result['labels'] = label['input_ids']

            return result


        # Use batched=True to activate fast multithreading!
        remove_columns = ['Abstract']
        if args.mask_stratagy == 'dynamic':
            remove_columns.append('Label')
        tokenized_datasets = esg_dataset.map(
            tokenize_function, batched=True, remove_columns = remove_columns
        )
        print(tokenized_datasets)



        chunk_size = args.chunk_size
        def group_texts(examples):
            # Concatenate all texts
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            # Compute length of concatenated texts
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the last chunk if it's smaller than chunk_size
            total_length = (total_length // chunk_size) * chunk_size
            # Split by chunks of max_len
            result = {
                k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
                for k, t in concatenated_examples.items()
            }
            # Create a new labels column
            if args.mask_stratagy != 'dynamic':
                result["labels"] = result["input_ids"].copy()

            return result



        lm_datasets = tokenized_datasets.map(group_texts, batched=True)
        print(lm_datasets)
        train_size = args.training_size
        test_size = int(0.1 * args.training_size)

        downsampled_dataset = lm_datasets["train"].train_test_split(
            train_size=train_size, test_size=test_size, seed=42
        )
        # downsampled_dataset.save_to_disk("esg-preprocessed")
        
    else:
        downsampled_dataset = load_from_disk(args.load_cache_dir)

    print(downsampled_dataset)

    from transformers import TrainingArguments

    batch_size = args.batch_size
    # Show the training loss with every epoch
    logging_steps = len(downsampled_dataset["train"]) // batch_size
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        push_to_hub=False,
        fp16=True,
        logging_steps=logging_steps,
        save_strategy="epoch",
        save_total_limit=1
    )

    from transformers import Trainer
    from transformers import DataCollatorForLanguageModeling



    if args.mask_stratagy == 'random':
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    elif args.mask_stratagy == 'dynamic':
        data_collator = DynamicDataCollatorForLanguageModeling(tokenizer=tokenizer)



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=downsampled_dataset["train"],
        eval_dataset=downsampled_dataset["test"],
        data_collator=data_collator,
    )

    
    # if args.do_eval:
    #     eval_results = trainer.evaluate()
    #     print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    if args.do_train:
        trainer.train()
    trainer.save_model()
    if args.do_eval:
        eval_results = trainer.evaluate()
        print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    import os
    os.system('shutdown -s')





if __name__ == "__main__":
    main()
