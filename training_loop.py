import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, AutoTokenizer, TrainingArguments
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from peft import LoraConfig
from trl import SFTTrainer
from transformers import  TrainingArguments, DataCollatorForLanguageModeling


def llama_inference(prompt):
    input_ = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
    output = model.generate(input, max_length=500)
    return tokenizer.decode(output[0])

def evaluate_model_on_dataset(dataset, num_rows=2):
    correct_count = 0

    # Initialize sentence transformer model
    #free embeddings!
    embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    for idx in range(num_rows):

        question = dataset['question'][idx]
        options = dataset['options'][idx]
        correct_answer_idx = dataset['answer_idx'][idx]
        answer = dataset['answer'][idx]

        # Run the model's inference on the medical question
        prompt = question
        response = llama_inference(prompt)
        print('Response:' + response)

        # Generate embeddings for the response and correct answer
        response_embedding = embedder.encode(response, convert_to_tensor=True)
        correct_answer_embedding = embedder.encode(answer, convert_to_tensor=True)

        # Compute cosine similarity
        cosine_similarity = util.pytorch_cos_sim(response_embedding, correct_answer_embedding).item()
        print('the similarity is ' + str(cosine_similarity))
        is_correct = cosine_similarity >= 0.3 # Adjust the threshold as needed, >30% threshold

        if is_correct:
            correct_count += 1

        print(f"Correct Answer: {answer}")
        print(f"Is Model's Response Correct? {is_correct}\n")

    accuracy = correct_count / num_rows * 100
    print(f"Accuracy on the first {num_rows} rows: {accuracy}%")

def preprocess_function(examples):
    return {
        "input_ids": tokenizer(examples["instruction"] + " " + examples["input"], truncation=True, max_length=512)["input_ids"],
        "labels": tokenizer(examples["output"], truncation=True, max_length=512)["input_ids"],
    }
     
scaler = torch.cuda.amp.GradScaler()

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype="float16", #halves the size of the mdoel
        bnb_4bit_use_double_quant=False,
    )
device_map = {"": 0}

model = AutoModelForCausalLM.from_pretrained(
        "saleem-ullah/Llama-2-7b-chat-hf",
        quantization_config=bnb_config,
        device_map=device_map,
        use_auth_token=True
    )

model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained('saleem-ullah/Llama-2-7b-chat-hf', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split='train')

datasets_names = [
    "medalpaca/medical_meadow_mediqa",
    "medalpaca/medical_meadow_mmmlu",
    "medalpaca/medical_meadow_medical_flashcards",
    "medalpaca/medical_meadow_wikidoc_patient_information",
    "medalpaca/medical_meadow_wikidoc",
    "medalpaca/medical_meadow_pubmed_causal",
    "medalpaca/medical_meadow_medqa",
    "medalpaca/medical_meadow_health_advice",
    "medalpaca/medical_meadow_cord19",
]

datasets = [load_dataset(name, split="train") for name in datasets_names]
combined_dataset = concatenate_datasets(datasets)
processed_dataset = combined_dataset.map(preprocess_function)

# Set fine tune params
training_arguments = TrainingArguments(
    output_dir='results/',
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim='paged_adamw_32bit',
    save_steps=5000,
    logging_steps=1000,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=5000,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type='constant',
)
model.config.use_cache = False

# Train loop
model.config.pretraining_tp = 1
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)


# Define data collator to handle tokenization and collation
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset,
    peft_config=peft_config,
    dataset_text_field="input",
    max_seq_length=512,
    args=training_arguments,
    data_collator=data_collator,
    packing=False,
)
trainer.train()

# Save model
model_save_path = './my_model'
trainer.save_model(model_save_path)
trainer.push_to_hub()
     
