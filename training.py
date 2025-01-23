
import os
import pandas as pd
import numpy as np
import pickle
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from imblearn.over_sampling import RandomOverSampler

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset


# 3. Duomenų kokybės analizė ir vizualizacija
file_paths = [
    "cleaned_and_lowercased_data.csv",
    "cleaned_data.csv",
    "lowercase_data.csv"
]
data_frames = [pd.read_csv(file_path, encoding='utf-8') for file_path in file_paths]
training_data = pd.concat(data_frames, ignore_index=True)

print(training_data.head())
print(training_data.info())

subcategory_counts = training_data['subcategory'].value_counts().sort_values(ascending=False)
print(subcategory_counts)


# 4. Duomenų valymas ir apdorojimas
def clean_text(text):
    text = str(text).lower()  # Konvertuokite į mažąsias raides
    text = re.sub(r'[^\w\s]', '', text)  # Pašalinkite skyrybos ženklus
    text = re.sub(r'\s+', ' ', text).strip()  # Pašalinkite perteklių tarpų
    return text

training_data['name'] = training_data['name'].apply(clean_text)



# 5. Duomenų padalijimas į treniravimo ir validacijos rinkinius
# Pašalinkite eilutes su trūkstamais duomenimis
training_data = training_data.dropna(subset=['name', 'subcategory'])

# Inicializuokite LabelEncoder ir transformuokite subkategorijas į label'us
label_encoder = LabelEncoder()
training_data['subcategory_label'] = label_encoder.fit_transform(training_data['subcategory'])

# Padalinkite duomenis į treniravimo ir validacijos rinkinius
train_df, val_df = train_test_split(
    training_data,
    test_size=0.1,
    stratify=training_data['subcategory_label'],
    random_state=42
)

print(train_df.shape)
print(val_df.shape)


# 6. LabelEncoder nustatymas ir tikrinimas
print("LabelEncoder klasės:", label_encoder.classes_)

# 7. Duomenų subalansavimas naudojant RandomOverSampler
ros = RandomOverSampler(random_state=42)

# Transformuokite treniravimo duomenis
X_train = train_df['name'].values.reshape(-1, 1)
y_train = train_df['subcategory_label'].values

X_res, y_res = ros.fit_resample(X_train, y_train)

# Sukurkite subalansuotą treniravimo DataFrame
train_df_balanced = pd.DataFrame({
    'name': X_res.flatten(),
    'subcategory_label': y_res
})

print(train_df_balanced['subcategory_label'].value_counts())


# 8. Duomenų tokenizavimas
def create_dataset(df):
    return Dataset.from_pandas(df[['name', 'subcategory_label']].rename(columns={'name': 'text', 'subcategory_label': 'label'}))

train_dataset = create_dataset(train_df_balanced)
val_dataset = create_dataset(val_df)



# Pasirinkite mažesnį modelį, pvz., DistilRoBERTa
model_name = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenizacijos funkcija su sumažintu max_length
def tokenize_with_labels(batch):
    encoding = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=32)  # Sumažinkite max_length
    encoding["labels"] = batch["label"]
    return encoding

# Tokenizuokite duomenis
train_dataset = train_dataset.map(tokenize_with_labels, batched=True)
val_dataset = val_dataset.map(tokenize_with_labels, batched=True)

# Pašalinkite nereikalingus stulpelius
train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]])
val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]])

# 9. Klasių svorių apskaičiavimas
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df_balanced['subcategory_label']),
    y=train_df_balanced['subcategory_label']
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Klasių svoriai:", class_weights)

# 10. Modelio treniravimas su validacijos rinkiniu
# Įkelkite modelį
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

# Apibrėžkite treniravimo argumentus su `report_to='none'`, `eval_strategy` ir `fp16`
training_args = TrainingArguments(
    output_dir='./model_output',
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    save_strategy="epoch",
    eval_strategy="epoch",  # Pakeista iš `evaluation_strategy`
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    overwrite_output_dir=True,
    learning_rate=2e-5,
    report_to='none',  # Išjungia wandb
    fp16=True,  # Įjungia mixed precision treniravimą
    gradient_accumulation_steps=2
)

# Apibrėžkite duomenų tvarkyklę
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Apibrėžkite metrikas
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = (predictions == labels).mean()
    return {"accuracy": acc}

# Sukurkite Trainer su klasių svoriu
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer.train()

# Išsaugokite modelį ir tokenizer
model.save_pretrained('./final_model-2')
tokenizer.save_pretrained('./final_model-2')

# (Pasirinktinai) Išsaugokite LabelEncoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)