import pandas as pd
import random

import os

class BookData:
    def __init__(self, csv_path):
        # Convert to absolute path if it's not already
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path)
        
        self.df = pd.read_csv(csv_path, on_bad_lines='skip')
        self._clean_data()
        
    def _clean_data(self):
        # Clean the data
        self.df = self.df.dropna(subset=['title', 'authors'])
        self.df['average_rating'] = pd.to_numeric(self.df['average_rating'], errors='coerce')
        self.df = self.df.dropna(subset=['average_rating'])
        
    def search_by_title(self, title):
        return self.df[self.df['title'].str.contains(title, case=False)].to_dict('records')
    
    def search_by_author(self, author):
        # Improved author search that handles name variations
        author = str(author).lower()
        # Search for author name in any part of the authors string
        mask = self.df['authors'].str.lower().str.contains(author, na=False)
        # Also try splitting multi-author entries
        if not mask.any():
            mask = self.df['authors'].str.lower().str.contains('|'.join(author.split()), na=False)
        return self.df[mask].to_dict('records')
    
    def search_by_rating(self, min_rating=4.0):
        return self.df[self.df['average_rating'] >= float(min_rating)].sample(5).to_dict('records')
    
    def search_by_language(self, language_code):
        return self.df[self.df['language_code'] == language_code.lower()].sample(5).to_dict('records')
    
    def get_random_book(self):
        return self.df.sample(1).to_dict('records')[0]
    
    def format_book_response(self, book):
        return f"Title: {book['title']}\nAuthor: {book['authors']}\nRating: {book['average_rating']}\nISBN: {book['isbn']}\nPages: {book['  num_pages']}"
    
    def format_multiple_books(self, books):
        if not books:
            return "Sorry, I couldn't find any matching books."
        
        response = "Here are some recommendations for you:\n\n"
        for i, book in enumerate(books[:5], 1):  # Limit to 5 books
            response += f"{i}. {book['title']} by {book['authors']} (Rating: {book['average_rating']})\n"
        return response