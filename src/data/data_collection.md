{'loss': 0.0002, 'grad_norm': 0.02192065119743347, 'learning_rate': 2.1344717182497334e-08, 'epoch': 1.0}
{'loss': 0.0001, 'grad_norm': 0.002364831045269966, 'learning_rate': 1.3135210573844512e-08, 'epoch': 1.0}
{'eval_loss': 0.03646375238895416, 'eval_runtime': 417.1397, 'eval_samples_per_second': 116.798, 'eval_steps_per_second': 14.602, 'epoch': 1.0}
{'train_runtime': 9534.9661, 'train_samples_per_second': 40.88, 'train_steps_per_second': 2.555, 'train_loss': 0.04835136267889097, 'epoch': 1.0}
100%|██████████████████████████████████████████████████████████████████████████| 24362/24362 [2:38:54<00:00,  2.56it/s]
2025-03-21 13:02:39,964: Model and tokenizer saved to ./models/distilroberta

(AI-detection) (base) C:\Users\talkt\Documents\GitHub\AI-generated-content-detection\src\models>python evaluate_model.py

100%|██████████████████████████████████████████████████████████████████████████████| 6091/6091 [11:28<00:00,  8.84it/s]
2025-03-21 13:26:16,510: Test Accuracy: 0.9927

(AI-detection) (base) C:\Users\talkt\Documents\GitHub\AI-generated-content-detection\src\models>

cd C:\Users\talkt\Documents\GitHub\AI-generated-content-detection
AI-detection\Scripts\activate
streamlit run src/frontend/app.py