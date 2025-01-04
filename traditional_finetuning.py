import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
from tqdm import tqdm
import time
from torch.nn import CrossEntropyLoss
from rouge_score import rouge_scorer
import time
from tqdm import tqdm

MODEL_NAME = "gpt2"
BATCH_SIZE = 1
EPOCHS = 1
MAX_LEN = 1024
GRADIENT_ACCUMULATION_STEPS = 4
GRADIENT_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 1
LEARNING_RATE = 1e-3

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna().sample(frac=0.1)

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    tokenized_articles = []
    tokenized_summaries = []
    for article, summary in zip(df["article"], df["highlights"]):
        max_length_article = MAX_LEN
        article_tokens = tokenizer.encode(article, truncation=True, max_length=max_length_article)
        summary_tokens = tokenizer.encode(summary, truncation=True, max_length=MAX_LEN)
        padded_article = article_tokens + [tokenizer.eos_token_id] * (max_length_article - len(article_tokens))
        padded_summary = summary_tokens + [tokenizer.eos_token_id] * (MAX_LEN - len(summary_tokens))
        tokenized_articles.append(padded_article)
        tokenized_summaries.append(padded_summary)

    return tokenized_articles, tokenized_summaries

def calculate_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(pred, ref) for pred, ref in zip(predictions, references)]
    avg_rouge1 = sum([s['rouge1'].fmeasure for s in scores]) / len(scores)
    avg_rouge2 = sum([s['rouge2'].fmeasure for s in scores]) / len(scores)
    avg_rougeL = sum([s['rougeL'].fmeasure for s in scores]) / len(scores)
    return avg_rouge1, avg_rouge2, avg_rougeL

def fine_tune_on_summarization(model, train_articles, train_summaries, val_articles, val_summaries):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id)
    best_val_loss = float('inf')
    no_improvement_epochs = 0
    total_start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_start_event = torch.cuda.Event(enable_timing=True)
        train_end_event = torch.cuda.Event(enable_timing=True)
        train_start_event.record()

        with tqdm(enumerate(zip(train_articles, train_summaries)), total=len(train_articles), desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch") as progress:
            for idx, (article, summary) in progress:
                input_ids = torch.tensor(article).to(device)
                labels = torch.tensor(summary).to(device)
                outputs = model(input_ids=input_ids)
                logits = outputs.logits
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad()

            avg_train_loss = train_loss / len(train_articles)
            print(f"Train Loss (Epoch {epoch + 1}): {avg_train_loss:.4f}")

        train_end_event.record()
        train_end_event.synchronize()
        train_gpu_time = train_start_event.elapsed_time(train_end_event)

        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_references = []
        val_start_event = torch.cuda.Event(enable_timing=True)
        val_end_event = torch.cuda.Event(enable_timing=True)
        val_start_event.record()

        with torch.no_grad():
            for article, summary in tqdm(zip(val_articles, val_summaries), total=len(val_articles), desc="Validation", unit="batch"):
                input_ids = torch.tensor(article).to(device)
                labels = torch.tensor(summary).to(device)
                outputs = model(input_ids=input_ids)
                logits = outputs.logits
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                val_loss += loss.item()
                predicted_token_ids = torch.argmax(logits, dim=-1)
                pred_text = tokenizer.decode(predicted_token_ids.squeeze(0), skip_special_tokens=True)
                ref_text = tokenizer.decode(labels, skip_special_tokens=True)
                val_predictions.append(pred_text)
                val_references.append(ref_text)

            avg_val_loss = val_loss / len(val_articles)
            avg_rouge1, avg_rouge2, avg_rougeL = calculate_rouge(val_predictions, val_references)
            print(f"Val Loss (Epoch {epoch + 1}): {avg_val_loss:.4f}")
            print(f"Val ROUGE-1: {avg_rouge1:.4f}, Val ROUGE-2: {avg_rouge2:.4f}, Val ROUGE-L: {avg_rougeL:.4f}")

        val_end_event.record()
        val_end_event.synchronize()
        val_gpu_time = val_start_event.elapsed_time(val_end_event)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
                break

        print(f"GPU Compute Time (Train Epoch {epoch + 1}): {train_gpu_time:.2f} ms")
        print(f"GPU Compute Time (Validation Epoch {epoch + 1}): {val_gpu_time:.2f} ms")
        
        torch.save(model.state_dict(), f"traditional_finetuning_{epoch}.pth")

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    return model

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenized_articles_train, tokenized_summaries_train = load_and_preprocess_data("/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/train.csv")
tokenized_articles_validation, tokenized_summaries_validation = load_and_preprocess_data("/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/validation.csv")
tokenized_articles_test, tokenized_summaries_test = load_and_preprocess_data("/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/test.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenized_articles_train = tokenized_articles_train[:21000]
tokenized_summaries_train = tokenized_summaries_train[:21000]
tokenized_articles_validation = tokenized_articles_validation[:6000]
tokenized_summaries_validation = tokenized_summaries_validation[:6000]
tokenized_articles_test = tokenized_articles_test[:3000]
tokenized_summaries_test = tokenized_summaries_test[:3000]

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)

for param in model.transformer.parameters():
    param.requires_grad = False
for param in model.lm_head.parameters():
    param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")

fine_tuned_model = fine_tune_on_summarization(model, tokenized_articles_train, tokenized_summaries_train, tokenized_articles_validation, tokenized_summaries_validation)
torch.save(fine_tuned_model.state_dict(), 'traditional_finetuning.pth')

## Evaluation
fine_tuned_model.eval()
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id)
test_predictions = []
test_references = []
test_loss = 0.0

with torch.no_grad():
    for article, summary in tqdm(zip(tokenized_articles_test, tokenized_summaries_test), total=len(tokenized_articles_test), desc="Testing", unit="batch"):
        input_ids = torch.tensor(article).to(device)
        labels = torch.tensor(summary).to(device)
        outputs = fine_tuned_model(input_ids=input_ids)
        logits = outputs.logits
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        test_loss += loss.item()
        predicted_token_ids = torch.argmax(logits, dim=-1)
        pred_text = tokenizer.decode(predicted_token_ids.squeeze(0), skip_special_tokens=True)
        ref_text = tokenizer.decode(labels, skip_special_tokens=True)
        test_predictions.append(pred_text)
        test_references.append(ref_text)

    avg_rouge1_test, avg_rouge2_test, avg_rougeL_test = calculate_rouge(test_predictions, test_references)
    print(f"Test ROUGE-1: {avg_rouge1_test:.4f}, Test ROUGE-2: {avg_rouge2_test:.4f}, Test ROUGE-L: {avg_rougeL_test:.4f}")

# model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
# model.load_state_dict(torch.load('traditional_finetuning.pth'))

# model.eval()
# criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id)
# test_predictions = []
# test_references = []
# test_loss=0.0

# with torch.no_grad():
#     for article, summary in tqdm(zip(tokenized_articles_test, tokenized_summaries_test), total=len(tokenized_articles_test), desc="Testing", unit="batch"):
#         input_ids = torch.tensor(article).to(device)
#         labels = torch.tensor(summary).to(device)
#         outputs = model(input_ids=input_ids)
#         logits = outputs.logits
#         loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
#         test_loss += loss.item()
#         predicted_token_ids = torch.argmax(logits, dim=-1)
#         pred_text = tokenizer.decode(predicted_token_ids.squeeze(0), skip_special_tokens=True)
#         ref_text = tokenizer.decode(labels, skip_special_tokens=True)
#         test_predictions.append(pred_text)
#         test_references.append(ref_text)

#     avg_rouge1_test, avg_rouge2_test, avg_rougeL_test = calculate_rouge(test_predictions, test_references)
#     print(f"Test ROUGE-1: {avg_rouge1_test:.4f}, Test ROUGE-2: {avg_rouge2_test:.4f}, Test ROUGE-L: {avg_rougeL_test:.4f}")

