from data.data_prep import prepare_shakespeare_data
from models.character_languge_model import CharacterLanguageModel
from training import Trainer, create_training_config

# 1. Prepare data
train_loader, val_loader, tokenizer, info = prepare_shakespeare_data(
    context_length=256,
    batch_size=32
)

# 2. Create model
model = CharacterLanguageModel(
    vocab_size=info['vocab_size'],
    d_model=128,
    num_layers=4,
    num_heads=4,
    d_latent=16,
    d_ff=512,
    max_seq_len=256,
    dropout=0.1
)

# 3. Create training config
config = create_training_config(
    num_epochs=20,
    learning_rate=3e-4,
    batch_size=32
)

# 4. Train
trainer = Trainer(model, train_loader, val_loader, tokenizer, config)
trainer.train()

# 5. Generate text
trainer.load_checkpoint('best_model.pt')
text = trainer.generate_sample("To be or not to be", max_tokens=200)
print(text)