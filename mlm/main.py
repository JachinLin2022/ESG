from datasets import load_dataset
import pandas as pd
import argparse
import log
import math

logger = log.get_logger('root')


class DynamicDataCollatorForLanguageModeling:
    pass

def main():
    parser = argparse.ArgumentParser(description="Command line interface for ESG")

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


    args = parser.parse_args()
    logger.info("Parameters: {}".format(args))

    # source = pd.read_csv("/home/linzhisheng/esg/source.csv")
    # source = source[source['Abstract'].notna()]
    # print(source)
    # source = source[['Abstract']]
    # source.to_csv("source_all", index=False)
    # exit(0)

    esg_dataset = load_dataset("csv", data_files=args.data_path)
    # esg_dataset  = esg_dataset.train_test_split(test_size=0.2, shuffle=True)
    print(esg_dataset)


    from transformers import AutoModelForMaskedLM

    model_checkpoint = args.model_name_or_path
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)



    print("start to tokenize")
    def tokenize_function(examples):
        result = tokenizer(examples["Abstract"])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result


    # Use batched=True to activate fast multithreading!
    tokenized_datasets = esg_dataset.map(
        tokenize_function, batched=True, remove_columns = ['Abstract']
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
        result["labels"] = result["input_ids"].copy()
        return result



    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    print(lm_datasets)


    train_size = args.training_size
    test_size = int(0.1 * args.training_size)

    downsampled_dataset = lm_datasets["train"].train_test_split(
        train_size=train_size, test_size=test_size, seed=42
    )
    print(downsampled_dataset)


    from transformers import TrainingArguments

    batch_size = args.batch_size
    # Show the training loss with every epoch
    logging_steps = len(downsampled_dataset["train"]) // batch_size
    model_name = model_checkpoint.split("/")[-1]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        push_to_hub=False,
        fp16=True,
        logging_steps=logging_steps,
        save_strategy="epoch",
        save_total_limit=3
    )

    from transformers import Trainer
    from transformers import DataCollatorForLanguageModeling



    if args.mask_stratagy == 'random':
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    elif args.mask_stratagy == 'dynamic':
        data_collator = DynamicDataCollatorForLanguageModeling()



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=downsampled_dataset["train"],
        eval_dataset=downsampled_dataset["test"],
        data_collator=data_collator,
    )

    
    if args.do_eval:
        eval_results = trainer.evaluate()
        print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    if args.do_train:
        trainer.train()

    if args.do_eval:
        eval_results = trainer.evaluate()
        print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.save_model()

    # from transformers import DataCollatorForLanguageModeling

    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    # def insert_random_mask(batch):
    #     features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    #     masked_inputs = data_collator(features)
    #     # Create a new "masked" column for each column in the dataset
    #     return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}


    # train_size = 5_000_000
    # test_size = int(0.1 * train_size)

    # downsampled_dataset = lm_datasets["train"].train_test_split(
    #     train_size=train_size, test_size=test_size, seed=42
    # )
    # print(downsampled_dataset)

    # downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
    # eval_dataset = downsampled_dataset["test"].map(
    #     insert_random_mask,
    #     batched=True,
    #     remove_columns=downsampled_dataset["test"].column_names,
    # )
    # eval_dataset = eval_dataset.rename_columns(
    #     {
    #         "masked_input_ids": "input_ids",
    #         "masked_attention_mask": "attention_mask",
    #         "masked_labels": "labels",
    #     }
    # )

    # from torch.utils.data import DataLoader
    # from transformers import default_data_collator

    # batch_size = 64
    # train_dataloader = DataLoader(
    #     downsampled_dataset["train"],
    #     shuffle=True,
    #     batch_size=batch_size,
    #     collate_fn=data_collator,
    # )
    # eval_dataloader = DataLoader(
    #     eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
    # )

    # model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    # from torch.optim import AdamW

    # optimizer = AdamW(model.parameters(), lr=5e-5)
    # from accelerate import Accelerator

    # accelerator = Accelerator()
    # model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    #     model, optimizer, train_dataloader, eval_dataloader
    # )
    # print(accelerator.device)

    # from transformers import get_scheduler

    # num_train_epochs = 3
    # num_update_steps_per_epoch = len(train_dataloader)
    # num_training_steps = num_train_epochs * num_update_steps_per_epoch

    # lr_scheduler = get_scheduler(
    #     "linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps,
    # )
    # model_name = "distilbert-base-uncased-finetuned-imdb-accelerate"
    # output_dir = model_name
    # from tqdm.auto import tqdm
    # import torch
    # import math

    # progress_bar = tqdm(range(num_training_steps))

    # for epoch in range(num_train_epochs):
    #     # Training
    #     model.train()
    #     for batch in train_dataloader:
    #         outputs = model(**batch)
    #         loss = outputs.loss
    #         accelerator.backward(loss)

    #         optimizer.step()
    #         lr_scheduler.step()
    #         optimizer.zero_grad()
    #         progress_bar.update(1)

    #     # Evaluation
    #     model.eval()
    #     losses = []
    #     for step, batch in enumerate(eval_dataloader):
    #         with torch.no_grad():
    #             outputs = model(**batch)

    #         loss = outputs.loss
    #         losses.append(accelerator.gather(loss.repeat(batch_size)))

    #     losses = torch.cat(losses)
    #     losses = losses[: len(eval_dataset)]
    #     try:
    #         perplexity = math.exp(torch.mean(losses))
    #     except OverflowError:
    #         perplexity = float("inf")

    #     print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    #     # Save and upload
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    #     if accelerator.is_main_process:
    #         tokenizer.save_pretrained(output_dir)
    #         # repo.push_to_hub(
    #         #     commit_message=f"Training in progress epoch {epoch}", blocking=False
    #         # )

if __name__ == "__main__":
    main()