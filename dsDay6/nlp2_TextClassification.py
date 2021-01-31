import pandas as pd

def load_data(csv_file, split=0.9):
    data = pd.read_csv(csv_file)
    train_data = data.sample(frac=1, random_state=7)

    texts = train_data.text.values
    labels = [{"POSITIVE":bool(y), "NEGATIVE":not bool(y)} for y in train_data.sentiment.values]

    split = int(len(train_data) * split)

    train_labels = [{"cats":label} for label in labels[split:]]
    val_labels = [{"cats":label} for label in labels[split:]]
    return texts[:split], train_labels, texts[split:], val_labels

train_texts, train_labels, val_texts, val_labels = load_data('../text_classification/yelp_ratings.csv')

print('Texts from the training data\n------')
print(train_texts[:2])
print('Labels from the training data\n------')
print(train_labels[:2])


import spacy
# Create an empty model
nlp = spacy.blank('en')
# Create the TextCategorizer with exclusive classes and "bow" architecture
textcat = nlp.create_pipe("textcat",
                    config={"exclusive_classes":True,
                            "architecture":"bow"})
# Add the TextCategorizer to the empty model
nlp.add_pipe(textcat)
textcat.add_label("NEGATIVE")
textcat.add_label("POSITIVE")

## Train Function
from spacy.util import minibatch
import random

def train(model, train_data, optimizer, batch_size=8):
    random.seed(1)
    losses={}
    random.shuffle(train_data)

    batches = minibatch(train_data, size=batch_size)
    for batch in batches:

        texts, labels = zip(*batch)
        model.update(texts, labels, sgd=optimizer, losses=losses)

    return losses

spacy.util.fix_random_seed(1)
random.seed(1)

optimizer = nlp.begin_training()
train_data = list(zip(train_texts, train_labels))
losses = train(nlp, train_data, optimizer)
print(losses['textcat'])
# 3.57

text = "This tea cup was full of holes. Do not recommend."
doc = nlp(text)
print(doc.cats)
# {'NEGATIVE': 0.3510996401309967, 'POSITIVE': 0.6489003896713257}

def predict(nlp, texts):
    docs = [nlp.tokenizer(text) for text in texts]
    # Use textcat to get the scores for each doc
    textcat = nlp.get_pipe('textcat')
    scores, _ = textcat.predict(docs)

    predicted_class = scores.argmax(axis=1)
    return predicted_class

# Sample prediction
texts = val_texts[34:38]
predictions = predict(nlp, texts)

for p,t in zip(predictions, texts):
    print(f"{textcat.labels[p]}: {t}\n")


# Evaluate the model
def evaluate(model, texts, labels):
    # Get predictions from textcat model
    predicted_class = predict(model, texts)
    # From labels, get the true class as a list of integers (POSITIVE -> 1, NEGATIVE -> 0)
    true_class = [int(each['cats']['POSITIVE']) for each in labels]

    correct_predictions = predicted_class == true_class
    accuracy = correct_predictions.mean()
    return accuracy

accuracy = evaluate(nlp, val_texts, val_labels)
print(f"Accuracy: {accuracy:.4f}")
# Accuracy: 0.7503

n_iters=5;
for i in range(n_iters):
    losses = train(nlp, train_data, optimizer)
    accuracy = evaluate(nlp, val_texts, val_labels)
    print(f"Loss: {losses['textcat']:.3f} \t Accuracy: {accuracy:.3f}")

# Loss: 2.392 	 Accuracy: 0.729
# Loss: 1.671 	 Accuracy: 0.684
# Loss: 1.242 	 Accuracy: 0.688
# Loss: 0.931 	 Accuracy: 0.687
# Loss: 0.721 	 Accuracy: 0.682
