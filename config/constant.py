from transformers import TrainingArguments


experiment_name ="Montra-Guest-Experience-Analytics"
classification_model_name = "montra-sentiment-model"
DAGSHUB_USERNAME = "victoroluikpe"
DAGSHUB_REPO = "Guest_Hotel_Senti_Intelligent_Analysis"

model_name = "distilbert-base-multilingual-cased"
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"    # prevent promopt hijacking the run


)