# models/robbert.py
import gc
import torch
import optuna
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    TrainerCallback
)

class GarbageCollectionCallback(TrainerCallback):
    """Clears GPU memory cache at the end of each evaluation/trial step to prevent OOM."""
    def on_evaluate(self, args, state, control, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def optimize_transformer_with_optuna(train_df, val_df, model_name='pdelobelle/robbert-2023-dutch-base'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load directly from parameters (no inner splitting)
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    
    def tokenize_func(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)
    
    train_tokenized = train_dataset.map(tokenize_func, batched=True)
    val_tokenized = val_dataset.map(tokenize_func, batched=True)
    
    def model_init(trial=None):
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir="./optuna_transformer_results",
        eval_strategy="epoch",            
        save_strategy="no",               
        disable_tqdm=True,
        logging_steps=50,
        weight_decay=0.01,
        fp16=torch.cuda.is_available()    
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}
    
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,              
        compute_metrics=compute_metrics,
        callbacks=[GarbageCollectionCallback()] 
    )
    
    def my_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 3),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1)
        }
    
    print("Launching Optuna search for Hugging Face Trainer...")
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=my_hp_space,
        n_trials=10, 
        compute_objective=lambda metrics: metrics["eval_accuracy"]
    )
    
    print("\n--- Best Transformer Hyperparameters Found ---")
    print(best_run.hyperparameters)
    return best_run.hyperparameters


def train_transformer(train_df, val_df, test_df, model_name, epochs, batch_size, lr, save_path, run_optuna=False):
    """
    Final training orchestrator. Uses train_df and val_df to optimize the final model weights 
    and checks performance on test_df to assess test generalization.
    """
    if run_optuna:
        best_hps = optimize_transformer_with_optuna(train_df, val_df, model_name=model_name)
        lr = best_hps.get("learning_rate", lr)
        epochs = best_hps.get("num_train_epochs", epochs)
        batch_size = best_hps.get("per_device_train_batch_size", batch_size)
        weight_decay = best_hps.get("weight_decay", 0.01)
    else:
        weight_decay = 0.01

    print(f"\nFinal training setup: LR={lr:.2e}, Epochs={epochs}, Batch Size={batch_size}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
    
    def tokenize_func(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)
    
    train_tokenized = train_dataset.map(tokenize_func, batched=True)
    val_tokenized = val_dataset.map(tokenize_func, batched=True)
    test_tokenized = test_dataset.map(tokenize_func, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=save_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        # Checkpoint selection strictly bound to val split metrics
        load_best_model_at_end=True,      
        metric_for_best_model="loss",
        fp16=torch.cuda.is_available(),
        save_total_limit=1
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized, # Validation set tracks checkpoints
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[GarbageCollectionCallback()]
    )
    
    print("Training final Transformer model...")
    trainer.train()
    
    # Unbiased evaluation on the test split
    print("\n--- Evaluating Model on Held-out Test Set ---")
    test_results = trainer.evaluate(eval_dataset=test_tokenized)
    print(test_results)
    
    print(f"Saving final model and tokenizer to: {save_path}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)