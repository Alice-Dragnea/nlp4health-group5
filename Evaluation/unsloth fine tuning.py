
# will delete this file since we're pretty much set with fine tuning but wanted to keep for 

# embedding model fine tuning example from https://enzokro.dev/blog/blog_post?fpath=blog%2F007_fine_tune_encoders%2Ffine_tune_encoders.ipynb

#https://colab.research.google.com/github/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb

#can the pov influence classification accuracy? "the patient should vs you should..." bc we could use a model to restructure responses to fit format

#how to use bert embeddings for classification?
import unsloth

# general imports
from pathlib import Path
import torch

# import the huggingface classes
from transformers import BertModel 
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, TaskType

# import the sentence transformers classes
import sentence_transformers
from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer, SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# Model Configuration
BASE_MODEL_ID = 'sentence-transformers/all-MiniLM-L12-v2'
BERT_MODEL = BertModel

# Maximum sequence length of this model
MAX_SEQ_LENGTH = 512
LOAD_IN_4BIT = True  # For QLoRA (4-bit quantization)

# LORA Configuration
LORA_R = 16          # Rank of the LORA matrices.
LORA_ALPHA = 32      # Rule of thumb: 2 * rank
LORA_DROPOUT = 0.0   # Dropout of 0 is best.
USE_RSLORA = False   # Rank-Stabilized LoRA if desired

# Target modules for adapters
LORA_TARGET_MODULES = ["query", "key", "value", "dense"]
LORA_EXCLUDE_MODULES = [] # put anything you want to skip here

# Prepare the ST model
def get_st_unsloth_wrapper(
        model,
        tokenizer,
        base_model_id=BASE_MODEL_ID,
        pooling_mode="mean",
        max_seq_length=MAX_SEQ_LENGTH,
        ):
    print("Initializing Sentence Transformer modules...")

    # 1. Create the Transformer module instance
    transformer_module = sentence_transformers.models.Transformer(
        model_name_or_path=base_model_id,
        max_seq_length=max_seq_length,
    )

    # 2. Replace the internal Hugging Face model with our LORA-patched Unsloth model
    transformer_module.auto_model = model
    transformer_module.tokenizer = tokenizer

    print(f"Manually assigned Unsloth LORA model to sentence_transformers.models.Transformer module.")

    # 3. Create the Pooling module
    hidden_size = model.config.hidden_size
    pooling_module = sentence_transformers.models.Pooling(
        word_embedding_dimension=hidden_size,
        pooling_mode=pooling_mode,
    )
    print(f"Using Pooling module with mode: {pooling_mode}")

    # 4. Add the Normalize module
    normalize_module = sentence_transformers.models.Normalize()
    modules = [transformer_module, pooling_module, normalize_module]

    # 5. Initialize SentenceTransformer with custom modules
    sbert_model = SentenceTransformer(modules=modules)

    print(f"SentenceTransformer wrapper created with custom modules.")
    return sbert_model

# wrap our unsloth model in Sentence Transformers
sbert_model = get_st_unsloth_wrapper(
        model,
        tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        base_model_id=BASE_MODEL_ID,
        pooling_mode="mean",
  )

# define the loss function
loss = MultipleNegativesRankingLoss(sbert_model)
print(f"Using loss: {type(loss).__name__}")

## Preparing all of our training arguments

# Training Configuration
NUM_TRAIN_EPOCHS = 4                # Start with 1 epochs
PER_DEVICE_TRAIN_BATCH_SIZE = 512   # Adjust based on GPU VRAM and MAX_SEQ_LENGTH.
PER_DEVICE_EVAL_BATCH_SIZE = 1024   # Can usually be higher than train batch size.
GRADIENT_ACCUMULATION_STEPS = 1     # Only for small cards

# don't repeat samples in the same batch given our loss
batch_sampler = BatchSamplers.NO_DUPLICATES if isinstance(loss, MultipleNegativesRankingLoss) else None

# set the output directory
OUTPUT_DIR = Path("finetuned_embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# evaluation and saving
EVAL_STEPS = 4           # evaluate every N steps
EVAL_STRATEGY = "steps"
SAVE_STEPS = 4           # save checkpoint every N steps
SAVE_STRATEGY = "steps"
SAVE_TOTAL_LIMIT = 2     # keep only the last N checkpoints
LOGGING_STEPS = 2        # log metrics every N steps

# load test dataset
test_dataset = load_dataset("json", data_files="test_dataset.json", split="train")
train_dataset = load_dataset("json", data_files="train_dataset.json", split="train")
corpus_dataset = concatenate_datasets([train_dataset, test_dataset])

# Convert the datasets to dictionaries
corpus = dict(
    zip(corpus_dataset["id"], corpus_dataset["positive"])
)  # Our corpus (cid => document)
queries = dict(
    zip(test_dataset["id"], test_dataset["anchor"])
)  # Our queries (qid => question)

# Create a mapping of relevant document (1 in our case) for each query
relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
for q_id in queries:
    relevant_docs[q_id] = [q_id]


evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    score_functions={"cosine": cos_sim},
    name="ir-eval"
)

print("Defining training arguments...")
args = SentenceTransformerTrainingArguments(
    # Core Training Parameters
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    optim=OPTIMIZER,
    batch_sampler=batch_sampler,
    fp16=FP16,
    bf16=BF16,
    tf32=True, # NOTE: gpu must support
    fp16_full_eval=True,
    # Evaluation and Saving
    eval_strategy=EVAL_STRATEGY,
    eval_steps=EVAL_STEPS,
    save_strategy=SAVE_STRATEGY,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=True if evaluator else False,
    metric_for_best_model="eval_ir-eval_cosine_ndcg@10" if evaluator and isinstance(evaluator, InformationRetrievalEvaluator) else None,
    greater_is_better=True,
    # Logging and Reporting
    logging_steps=LOGGING_STEPS,
    report_to="tensorboard",
    run_name=f"{BASE_MODEL_ID.split('/')[-1]}-st-finetune",
    seed=42,
)

print("Initializing SentenceTransformerTrainer...")
trainer = SentenceTransformerTrainer(
    model=sbert_model, # Pass the standard SentenceTransformer model
    args=args,
    train_dataset=train_dataset.select_columns(["anchor", "positive"]),
    eval_dataset=test_dataset.select_columns(["anchor", "positive"]) if evaluator else None,
    loss=loss,
    evaluator=evaluator,
    callbacks=[],
)

# train the unsloth embeddings
train_res = trainer.train()

# save the fine-tuned for persistence
sbert_model.save_pretrained(str(OUTPUT_DIR))



## evaluation

# load the baseline
original_model = SentenceTransformer(BASE_MODEL_ID)
original_model.eval()

# load the fine-tuned model
fine_tuned_model = SentenceTransformer(
    args.output_dir, device="cuda" if torch.cuda.is_available() else "cpu"
)
fine_tuned_model.eval()

# evaluate both models
with torch.inference_mode():
  baselines = evaluator(original_model)
  fine_tuned_results = evaluator(fine_tuned_model)

# print their scores
print(f"Original model: {baselines}")
print(f"Fine-tuned model: {fine_tuned_results}")

# baseline result
baselines['ir-eval_cosine_ndcg@10']

# fine-tuned results
fine_tuned_results['ir-eval_cosine_ndcg@10']

# how much better is the fine-tune?
fine_tuned_results['ir-eval_cosine_ndcg@10'] / baselines['ir-eval_cosine_ndcg@10']