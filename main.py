# =============================================================================
# STEP 1 — SCRAPE ALL ANSES PUBLICATION PAGES (Human Nutrition Committee)
# =============================================================================
# Retrieves all paginated pages listing nutrition-related articles on the ANSES
# website. Pages range from 0 to ~90. Stops when no more results are found.
# =============================================================================

import re
import json
import torch
import tiktoken
import requests
import PyPDF2

from io import BytesIO
from typing import List
from bs4 import BeautifulSoup
from datasets import Dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- Configuration ---

BASE_URL = (
    "https://www.anses.fr/fr/content/avis-et-rapports-de-lanses-sur-saisine"
    "?field_title_value=&field_expert_committee=40005"
    "&field_referral_value=&field_linked_referrals_value="
    "&field_signature_date_value=&field_online_date_value="
    "&field_keywords_value=&field_thematique_target_id="
    "&sort_by=created"
)

OUTPUT_JSON_PATH = "articles_anses.json"
CHUNK_TARGET_TOKENS = 200
TOKENIZER_MODEL = "gpt-4o-mini"


# =============================================================================
# STEP 2 — TEXT CHUNKING UTILITIES
# =============================================================================

def split_text_on_sentence_boundaries(text: str) -> List[str]:
    """
    Splits a text into smaller units using '. ' and ';' as delimiters,
    preserving the punctuation at the end of each unit.
    """
    delimiter_pattern = r'(?<=[.;])\s+'
    raw_units = re.split(delimiter_pattern, text)
    cleaned_units = [unit.strip() for unit in raw_units if unit.strip()]
    return cleaned_units


def count_tokens_in_text(text: str, model: str = TOKENIZER_MODEL) -> int:
    """
    Returns the number of tokens in a text string using the tokenizer
    corresponding to the given model name.
    """
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))


def build_token_aware_chunks(
    text: str,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    model: str = TOKENIZER_MODEL
) -> List[str]:
    """
    Splits a text into chunks of approximately `target_tokens` tokens each,
    always cutting at sentence or clause boundaries (never mid-sentence).

    Args:
        text: The full text to chunk.
        target_tokens: Approximate maximum token count per chunk.
        model: The tokenizer model to use for counting tokens.

    Returns:
        A list of text chunks.
    """
    sentence_units = split_text_on_sentence_boundaries(text)
    chunks = []

    current_chunk = ""
    current_token_count = 0

    for unit in sentence_units:
        unit_token_count = count_tokens_in_text(unit, model)

        if current_token_count + unit_token_count <= target_tokens:
            # Unit fits in the current chunk
            current_chunk += (" " if current_chunk else "") + unit
            current_token_count += unit_token_count
        else:
            # Current chunk is full — save it and start a new one
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = unit
            current_token_count = unit_token_count

    # Append the last remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# =============================================================================
# STEP 3 — SCRAPE PAGINATED ARTICLE LISTING
# =============================================================================

def collect_all_listing_page_urls(base_url: str) -> List[str]:
    """
    Iterates over all paginated listing pages of the ANSES publication portal
    and returns a list of valid page URLs (stops when no results are found).
    """
    listing_page_urls = []
    page_index = 0

    while True:
        paginated_url = f"{base_url}&page={page_index}"
        response = requests.get(paginated_url)
        parsed_page = BeautifulSoup(response.text, "html.parser")

        if "Pas de document trouvé." in parsed_page.get_text():
            break

        listing_page_urls.append(paginated_url)
        page_index += 1

    return listing_page_urls


# =============================================================================
# STEP 4 — EXTRACT ARTICLE DATA FROM EACH LISTING PAGE
# =============================================================================

def parse_article_metadata(raw_metadata_element) -> dict:
    """
    Extracts structured metadata from an article's metadata HTML block.

    Returns a dictionary with keys:
        Type, Sous-titre, Comité d'experts, Numéro de saisine,
        Numéros de saisines liées, Date signature, Date de mise en ligne,
        Mots clés
    """
    metadata_fields = {
        "Type": None,
        "Sous-titre": None,
        "Comité d'experts": None,
        "Numéro de saisine": None,
        "Numéros de saisines liées": None,
        "Date signature": None,
        "Date de mise en ligne": None,
        "Mots clés": None,
    }

    raw_text = raw_metadata_element.get_text()
    text_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    for field_name in metadata_fields:
        if field_name in text_lines:
            field_index = text_lines.index(field_name)
            metadata_fields[field_name] = text_lines[field_index + 1]

    return metadata_fields


def extract_text_from_pdf_url(pdf_url: str) -> str:
    """
    Downloads a PDF from a URL and extracts its full text content
    by concatenating all pages (newlines removed).
    """
    response = requests.get(pdf_url)
    response.raise_for_status()

    pdf_reader = PyPDF2.PdfReader(BytesIO(response.content))
    full_text = ""

    for page in pdf_reader.pages:
        full_text += page.extract_text().replace("\n", "")

    return full_text


def scrape_all_articles(listing_page_urls: List[str]) -> List[dict]:
    """
    Iterates over all listing pages, visits each article's detail URL,
    extracts metadata and PDF text, and returns a structured list of
    article dictionaries.

    Each article dict contains:
        - Id: placeholder (not used)
        - Titre: article title
        - url: direct URL to the article page
        - metadonnees: parsed metadata dict
        - texte: list of token-aware text chunks from the PDF
    """
    all_articles = []
    page_counter = 1

    for listing_url in listing_page_urls:
        print(f"Processing listing page: {page_counter}")
        page_counter += 1

        # Skip known problematic page (page 72, index 72)
        if page_counter == 73:
            continue

        page_response = requests.get(listing_url)
        parsed_listing = BeautifulSoup(page_response.text, "html.parser")
        article_elements = parsed_listing.select("article.document-document-list")

        for article_element in article_elements:
            metadata_block = article_element.select("div.node-content")[0]
            title_block = article_element.select("h3.know-more__title")[0]

            article_data = {
                "Id": None,
                "Titre": None,
                "url": None,
                "metadonnees": None,
                "texte": None,
            }

            # Parse metadata
            article_data["metadonnees"] = parse_article_metadata(metadata_block)

            # Parse title and article URL
            article_data["url"] = "https://www.anses.fr" + title_block.find("a").get("href")
            article_data["Titre"] = title_block.find("span").get_text()

            # Skip non-PDF or Excel documents
            article_url = article_data["url"]
            if "xls" in article_url or ".pdf" not in article_url:
                continue

            # Extract and chunk PDF text
            pdf_text = extract_text_from_pdf_url(article_url)
            article_data["texte"] = build_token_aware_chunks(pdf_text)

            all_articles.append(article_data)

    return all_articles


# =============================================================================
# STEP 5 — RUN SCRAPING PIPELINE & SAVE TO JSON
# =============================================================================

listing_page_urls = collect_all_listing_page_urls(BASE_URL)
all_articles = scrape_all_articles(listing_page_urls)

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as output_file:
    json.dump(all_articles, output_file, indent=4, ensure_ascii=False)

print(f"JSON corpus saved to: {OUTPUT_JSON_PATH}")
print(f"Total articles collected: {len(all_articles)}")


# =============================================================================
# STEP 6 — QLORA FINE-TUNING PIPELINE
# =============================================================================
# Fine-tunes TinyLlama-1.1B-Chat using QLoRA (LoRA + 4-bit quantization)
# on the ANSES nutrition corpus.
#
# Install dependencies (Google Colab):
#   !pip install -q -U transformers datasets accelerate peft bitsandbytes
# =============================================================================

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

# --- Model configuration ---

BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_OUTPUT_DIR = "./tinyllama-anses-lora"
MAX_SEQUENCE_LENGTH = 512
TRAIN_TEST_SPLIT_RATIO = 0.05
MIN_CHUNK_CHARACTER_LENGTH = 50

# --- LoRA hyperparameters ---
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# --- Training hyperparameters ---
TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 4
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0


# =============================================================================
# STEP 7 — BUILD INSTRUCTION-FOLLOWING DATASET
# =============================================================================

def build_instruction_dataset(articles: List[dict], max_articles: int = None) -> Dataset:
    """
    Builds a supervised instruction-following dataset from article chunks.

    Each training example uses the following prompt format:
        System: Tu es un assistant expert en nutrition humaine.
        User: Réponds uniquement à partir du contexte ci-dessous.
              Contexte: {chunk}
              Question: Que dit ce passage ?
        Assistant: {chunk}

    Args:
        articles: List of article dicts (with 'texte' key containing chunks).
        max_articles: If set, only use the first N articles.

    Returns:
        A HuggingFace Dataset with 'instruction' and 'output' columns,
        split into train and test sets.
    """
    article_subset = articles[:max_articles] if max_articles else articles
    training_examples = []

    for article in article_subset:
        for text_chunk in article.get("texte", []):
            if len(text_chunk.strip()) < MIN_CHUNK_CHARACTER_LENGTH:
                continue

            instruction_prompt = (
                f"Réponds uniquement à partir du contexte ci-dessous.\n\n"
                f"Contexte :\n{text_chunk}\n\n"
                f"Question :\nQue dit ce passage ?"
            )

            training_examples.append({
                "instruction": instruction_prompt.strip(),
                "output": text_chunk.strip(),
            })

    print(f"Total training examples built: {len(training_examples)}")

    raw_dataset = Dataset.from_list(training_examples)
    split_dataset = raw_dataset.train_test_split(
        test_size=TRAIN_TEST_SPLIT_RATIO,
        shuffle=False
    )
    return split_dataset


# =============================================================================
# STEP 8 — TOKENIZATION
# =============================================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_training_example(example: dict) -> dict:
    """
    Formats a training example using TinyLlama's chat template and tokenizes it.
    Labels are set equal to input_ids (standard causal LM objective).

    Args:
        example: Dict with 'instruction' and 'output' keys.

    Returns:
        Tokenized dict with 'input_ids', 'attention_mask', and 'labels'.
    """
    chat_messages = [
        {"role": "system", "content": "Tu es un assistant expert en nutrition humaine."},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]

    formatted_chat = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    tokenized_output = tokenizer(
        formatted_chat,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        padding="max_length",
    )

    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output


# =============================================================================
# STEP 9 — LOAD MODEL WITH 4-BIT QUANTIZATION
# =============================================================================

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
)

base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)


# =============================================================================
# STEP 10 — INJECT LORA ADAPTERS
# =============================================================================

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()


# =============================================================================
# STEP 11 — BUILD DATASET & TOKENIZE
# =============================================================================

with open(OUTPUT_JSON_PATH, "r", encoding="utf-8") as corpus_file:
    loaded_articles = json.load(corpus_file)

# Note: only using first 5 articles due to GPU constraints (Tesla T4 on Colab)
split_dataset = build_instruction_dataset(loaded_articles, max_articles=5)

tokenized_dataset = split_dataset.map(
    tokenize_training_example,
    remove_columns=split_dataset["train"].column_names,
)


# =============================================================================
# STEP 12 — TRAINING
# =============================================================================

training_arguments = TrainingArguments(
    output_dir=LORA_OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    logging_steps=10,
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="linear",
    warmup_ratio=WARMUP_RATIO,
    max_grad_norm=MAX_GRAD_NORM,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=peft_model,
    args=training_arguments,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

trainer.train()


# =============================================================================
# STEP 13 — INFERENCE
# =============================================================================

def generate_response(user_prompt: str, max_new_tokens: int = 200) -> str:
    """
    Generates a response from the fine-tuned model given a user prompt.
    Uses TinyLlama's chat template to format the input correctly.

    Args:
        user_prompt: The user's question or instruction.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        The decoded model output as a string.
    """
    peft_model.eval()

    chat_messages = [
        {"role": "system", "content": "Tu es un assistant expert en nutrition humaine."},
        {"role": "user", "content": user_prompt},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(formatted_prompt, return_tensors="pt").to(peft_model.device)

    generated_ids = peft_model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        do_sample=True,
    )

    decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return decoded_output


# --- Example inference call ---

test_context = (
    "Les gousses, contenant 5 à 20 graines, se forment à mesure que la plante mûrit. "
    "La plante est récoltée entière et séchée au soleil. Les gousses sont ensuite battues "
    "et les graines sont utilisées entières ou moulues. Les graines sont utilisées comme "
    "épice pour rehausser la saveur et la couleur des aliments. Elles sont aussi utilisées "
    "à des fins médicinales dans les médecines traditionnelles européenne, ayurvédique et "
    "chinoise, notamment comme hypoglycémiant, anti-inflammatoire local, diurétique, "
    "emménagogue, antitussif… (OMS 2007)."
)

test_prompt = (
    f"Réponds uniquement à partir du contexte ci-dessous.\n\n"
    f"Contexte :\n{test_context}\n\n"
    f"Question :\nQue dit ce passage ?"
)

print(generate_response(test_prompt))
