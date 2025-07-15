import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from book_data import BookData
import os

# Initialize NLP components
lemmatizer = WordNetLemmatizer()
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load chatbot training data
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Initialize book data
book_data = BookData(os.path.join(script_dir, 'books.csv'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def extract_entity(sentence, entity_name):
    words = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(words)
    
    if entity_name == 'author':
        # Improved author extraction
        author_phrases = []
        current_phrase = []
        
        # Look for patterns indicating author
        trigger_words = ['by', 'author', 'written', 'from', 'karya', 'penulis']
        
        for i, (word, tag) in enumerate(tagged):
            if word.lower() in trigger_words:
                # Collect all proper nouns after trigger word
                for j in range(i+1, len(tagged)):
                    if tagged[j][1] in ['NNP', 'NNPS']:  # Proper nouns
                        current_phrase.append(tagged[j][0])
                    else:
                        if current_phrase:
                            author_phrases.append(' '.join(current_phrase))
                            current_phrase = []
                        break
        
        if current_phrase:  # Add any remaining phrase
            author_phrases.append(' '.join(current_phrase))
        
        return author_phrases[0] if author_phrases else None
    elif entity_name == 'author':
        # Look for patterns like 'by X', 'author X'
        for i, (word, tag) in enumerate(tagged[:-1]):
            if word.lower() in ['by', 'author', 'written', 'from']:
                author = []
                for j in range(i+1, len(tagged)):
                    if tagged[j][1] in ['NNP', 'NNPS']:  # Proper nouns
                        author.append(tagged[j][0])
                if author:
                    return ' '.join(author)
    
    elif entity_name == 'min_rating':
        # Find rating numbers
        for word, tag in tagged:
            if tag == 'CD' and word.replace('.', '').isdigit():
                return float(word)
        return 4.0  # Default
    
    elif entity_name == 'language':
        # Find language
        lang_map = {
            'english': 'eng',
            'indonesian': 'ind',
            'spanish': 'spa',
            'french': 'fre',
            'german': 'ger'
        }
        for word in words:
            if word.lower() in lang_map:
                return lang_map[word.lower()]
        return 'eng'  # Default
    
    return None

def get_response(intents_list, intents_json, user_message):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for intent in list_of_intents:
        if intent['tag'] == tag:
            # Handle special actions
            if 'action' in intent:
                if intent['action'] == 'search_db':
                    # Extract required entities
                    entities = {}
                    for entity_type in intent['context']:
                        entities[entity_type] = extract_entity(user_message, entity_type)
                    
                    # Perform search based on entities
                    if 'title' in entities and entities['title']:
                        books = book_data.search_by_title(entities['title'])
                        response = book_data.format_multiple_books(books)
                    elif 'author' in entities and entities['author']:
                        books = book_data.search_by_author(entities['author'])
                        response = book_data.format_multiple_books(books)
                    elif 'min_rating' in entities:
                        books = book_data.search_by_rating(entities['min_rating'])
                        response = book_data.format_multiple_books(books)
                    elif 'language' in entities:
                        books = book_data.search_by_language(entities['language'])
                        response = book_data.format_multiple_books(books)
                    else:
                        response = "Sorry, I didn't understand your search criteria."
                    
                    return response
                
                elif intent['action'] == 'fetch_random':
                    book = book_data.get_random_book()
                    return book_data.format_book_response(book)
            
            # If no special action, return random response
            return random.choice(intent['responses'])
    
    return "Sorry, I didn't understand that. Could you please rephrase?"

print("Book Recommendation Bot is running!")
print("Ask me about books, for example:")
print("- Search for 'Harry Potter'")
print("- Books by J.K. Rowling")
print("- Highly rated books")
print("- Books in Spanish")

while True:
    try:
        message = input("You: ")
        if message.lower() in ['quit', 'exit', 'bye']:
            break
            
        ints = predict_class(message)
        res = get_response(ints, intents, message)
        print("Bot:", res)
    except KeyboardInterrupt:
        print("\nBot stopped.")
        break
    except Exception as e:
        print("Bot: Sorry, something went wrong. Please try again.")
        print(f"Error: {e}")