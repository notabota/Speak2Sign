import argparse
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_model", default="t5-small")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    tok = T5Tokenizer.from_pretrained(args.base_model)
    model = T5ForConditionalGeneration.from_pretrained(args.base_model)

    def encode(example):
        model_in = tok(example["input"], truncation=True, max_length=256)
        with tok.as_target_tokenizer():
            model_out = tok(example["target"], truncation=True, max_length=128)
        model_in["labels"] = model_out["input_ids"]
        return model_in

    train_ds = load_dataset("json", data_files=args.train, split="train").map(encode, batched=False)
    dev_ds   = load_dataset("json", data_files=args.dev,   split="train").map(encode, batched=False)

    collator = DataCollatorForSeq2Seq(tok, model=model)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=dev_ds, data_collator=collator, tokenizer=tok)
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
