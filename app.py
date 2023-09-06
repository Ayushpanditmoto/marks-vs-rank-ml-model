# Import libraries
from flask import Flask, request, jsonify
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download NLTK resources
nltk.download()


# Load PDF text


def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_text += page.extract_text()
    return pdf_text

# Preprocess text


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Stemming
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])

    return text

# Define a synthetic labeled dataset


def create_labeled_dataset():
    dataset = [
        {"text": "Medical journal article on heart disease.", "category": "medical"},
        {"text": "Financial report for XYZ Corporation.", "category": "finance"},
        {"text": "Travel itinerary for a business trip.", "category": "travel"},
        {"text": "Official government tax form.", "category": "official"},
        {"text": "Legal contract for a real estate transaction.", "category": "legal"},
        {"text": "Sports news coverage of the World Cup.", "category": "sports"},
        {"text": "Entertainment review of a new movie release.",
            "category": "entertainment"},
        {"text": "Scientific research paper on climate change.", "category": "science"},
        {"text": "History textbook on ancient civilizations.", "category": "education"},
        {"text": "Cooking recipe for a classic lasagna.", "category": "food"},
        {"text": "Technology news article about AI advancements.",
            "category": "technology"},
        {"text": "Environmental report on pollution levels.",
            "category": "environment"},
        {"text": "Travel guidebook for a European tour.", "category": "travel"},
        {"text": "Legal case brief for a high-profile lawsuit.", "category": "legal"},
        {"text": "Sports analysis of a championship game.", "category": "sports"},
        {"text": "Fashion magazine article on latest trends.", "category": "fashion"},
        {"text": "Health and wellness blog post.", "category": "health"},
        {"text": "Political analysis of current events.", "category": "politics"},
        {"text": "Entertainment interview with a celebrity.",
            "category": "entertainment"},
        {"text": "Financial guide for personal budgeting.", "category": "finance"},
        {"text": "Travel blog about backpacking adventures.", "category": "travel"},
        {"text": "Medical research paper on cancer treatments.", "category": "medical"},
        {"text": "Technology product review for a smartphone.",
            "category": "technology"},
        {"text": "History book on the American Revolution.", "category": "education"},
        {"text": "Legal document for a will and testament.", "category": "legal"},
        {"text": "Sports commentary on a football match.", "category": "sports"},
        {"text": "Environmental impact assessment report.", "category": "environment"},
        {"text": "Fashion tips and style recommendations.", "category": "fashion"},
        {"text": "Health and fitness magazine article.", "category": "health"},
        {"text": "Medical journal article on heart disease.", "category": "medical"},
        {"text": "Financial report for XYZ Corporation.", "category": "finance"},
        {"text": "Travel itinerary for a business trip.", "category": "travel"},
        {"text": "Official government tax form.", "category": "official"},
        {"text": "Legal contract for a real estate transaction.", "category": "legal"},
        {"text": "Sports news coverage of the World Cup.", "category": "sports"},
        {"text": "Entertainment review of a new movie release.",
            "category": "entertainment"},
        {"text": "Scientific research paper on climate change.", "category": "science"},
        {"text": "History textbook on ancient civilizations.", "category": "education"},
        {"text": "Cooking recipe for a classic lasagna.", "category": "food"},
        {"text": "Technology news article about AI advancements.",
            "category": "technology"},
        {"text": "Environmental report on pollution levels.",
            "category": "environment"},
        {"text": "Travel guidebook for a European tour.", "category": "travel"},
        {"text": "Legal case brief for a high-profile lawsuit.", "category": "legal"},
        {"text": "Sports analysis of a championship game.", "category": "sports"},
        {"text": "Fashion magazine article on latest trends.", "category": "fashion"},
        {"text": "Health and wellness blog post.", "category": "health"},
        {"text": "Political analysis of current events.", "category": "politics"},
        {"text": "Entertainment interview with a celebrity.",
            "category": "entertainment"},
        {"text": "Financial guide for personal budgeting.", "category": "finance"},
        {"text": "Travel blog about backpacking adventures.", "category": "travel"},
        {"text": "Medical research paper on cancer treatments.", "category": "medical"},
        {"text": "Technology product review for a smartphone.",
            "category": "technology"},
        {"text": "History book on the American Revolution.", "category": "education"},
        {"text": "Legal document for a will and testament.", "category": "legal"},
        {"text": "Sports commentary on a football match.", "category": "sports"},
        {"text": "Environmental impact assessment report.", "category": "environment"},
        {"text": "Fashion tips and style recommendations.", "category": "fashion"},
        {"text": "Health and fitness magazine article.", "category": "health"},
        {"text": "Political opinion piece on recent elections.", "category": "politics"},
        {"text": "Music review of a new album release.", "category": "entertainment"},
        {"text": "Investment guide for beginners.", "category": "finance"},
        {"text": "Travel diary of a backpacker in Asia.", "category": "travel"},
        {"text": "Medical case study on diabetes treatment.", "category": "medical"},
        {"text": "Technology trends report for the year.", "category": "technology"},
        {"text": "Geography textbook on world continents.", "category": "education"},
        {"text": "Legal contract for business partnership.", "category": "legal"},
        {"text": "Sports interview with an athlete.", "category": "sports"},
        {"text": "Environmental policy document.", "category": "environment"},
        {"text": "Fashion magazine feature on haute couture.", "category": "fashion"},
        {"text": "Guidelines and Process flow for the event.Registeration and Team Formation. ",
            "category": "education"},
        {"text": "Healthy eating tips and recipes.", "category": "health"}
        # Add more documents and categories as needed
    ]
    return dataset
# Main function for document classification


def classify_document(synthetic_dataset, pdf_path):
    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    preprocessed_pdf_text = preprocess_text(pdf_text)

    # Create synthetic labeled dataset
    X = [entry["text"] for entry in synthetic_dataset]
    y = [entry["category"] for entry in synthetic_dataset]

    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)
    pdf_text_vec = vectorizer.transform([preprocessed_pdf_text])

    # Train a Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_vec, y)

    # Make predictions on the preprocessed PDF text
    predicted_category = classifier.predict(pdf_text_vec)

    return predicted_category[0]


# Example usage
synthetic_dataset = create_labeled_dataset()
pdf_path = "demo.pdf"
predicted_category = classify_document(synthetic_dataset, pdf_path)
print("Predicted category:", predicted_category)


app = Flask(__name__)

# Download NLTK resources
nltk.download('stopwords')

# Load PDF text


def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_text += page.extract_text()
    return pdf_text
    # Your existing code for extracting text from a PDF

# Preprocess text


def preprocess_text(text):
    text = text.lower()

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Stemming
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])

    return text
    # Your existing code for text preprocessing

# Define a synthetic labeled dataset


def create_labeled_dataset():
    dataset = [
        {"text": "Medical journal article on heart disease.", "category": "medical"},
        {"text": "Financial report for XYZ Corporation.", "category": "finance"},
        {"text": "Travel itinerary for a business trip.", "category": "travel"},
        {"text": "Official government tax form.", "category": "official"},
        {"text": "Legal contract for a real estate transaction.", "category": "legal"},
        {"text": "Sports news coverage of the World Cup.", "category": "sports"},
        {"text": "Entertainment review of a new movie release.",
            "category": "entertainment"},
        {"text": "Scientific research paper on climate change.", "category": "science"},
        {"text": "History textbook on ancient civilizations.", "category": "education"},
        {"text": "Cooking recipe for a classic lasagna.", "category": "food"},
        {"text": "Technology news article about AI advancements.",
            "category": "technology"},
        {"text": "Environmental report on pollution levels.",
            "category": "environment"},
        {"text": "Travel guidebook for a European tour.", "category": "travel"},
        {"text": "Legal case brief for a high-profile lawsuit.", "category": "legal"},
        {"text": "Sports analysis of a championship game.", "category": "sports"},
        {"text": "Fashion magazine article on latest trends.", "category": "fashion"},
        {"text": "Health and wellness blog post.", "category": "health"},
        {"text": "Political analysis of current events.", "category": "politics"},
        {"text": "Entertainment interview with a celebrity.",
            "category": "entertainment"},
        {"text": "Financial guide for personal budgeting.", "category": "finance"},
        {"text": "Travel blog about backpacking adventures.", "category": "travel"},
        {"text": "Medical research paper on cancer treatments.", "category": "medical"},
        {"text": "Technology product review for a smartphone.",
            "category": "technology"},
        {"text": "History book on the American Revolution.", "category": "education"},
        {"text": "Legal document for a will and testament.", "category": "legal"},
        {"text": "Sports commentary on a football match.", "category": "sports"},
        {"text": "Environmental impact assessment report.", "category": "environment"},
        {"text": "Fashion tips and style recommendations.", "category": "fashion"},
        {"text": "Health and fitness magazine article.", "category": "health"},
        {"text": "Medical journal article on heart disease.", "category": "medical"},
        {"text": "Financial report for XYZ Corporation.", "category": "finance"},
        {"text": "Travel itinerary for a business trip.", "category": "travel"},
        {"text": "Official government tax form.", "category": "official"},
        {"text": "Legal contract for a real estate transaction.", "category": "legal"},
        {"text": "Sports news coverage of the World Cup.", "category": "sports"},
        {"text": "Entertainment review of a new movie release.",
            "category": "entertainment"},
        {"text": "Scientific research paper on climate change.", "category": "science"},
        {"text": "History textbook on ancient civilizations.", "category": "education"},
        {"text": "Cooking recipe for a classic lasagna.", "category": "food"},
        {"text": "Technology news article about AI advancements.",
            "category": "technology"},
        {"text": "Environmental report on pollution levels.",
            "category": "environment"},
        {"text": "Travel guidebook for a European tour.", "category": "travel"},
        {"text": "Legal case brief for a high-profile lawsuit.", "category": "legal"},
        {"text": "Sports analysis of a championship game.", "category": "sports"},
        {"text": "Fashion magazine article on latest trends.", "category": "fashion"},
        {"text": "Health and wellness blog post.", "category": "health"},
        {"text": "Political analysis of current events.", "category": "politics"},
        {"text": "Entertainment interview with a celebrity.",
            "category": "entertainment"},
        {"text": "Financial guide for personal budgeting.", "category": "finance"},
        {"text": "Travel blog about backpacking adventures.", "category": "travel"},
        {"text": "Medical research paper on cancer treatments.", "category": "medical"},
        {"text": "Technology product review for a smartphone.",
            "category": "technology"},
        {"text": "History book on the American Revolution.", "category": "education"},
        {"text": "Legal document for a will and testament.", "category": "legal"},
        {"text": "Sports commentary on a football match.", "category": "sports"},
        {"text": "Environmental impact assessment report.", "category": "environment"},
        {"text": "Fashion tips and style recommendations.", "category": "fashion"},
        {"text": "Health and fitness magazine article.", "category": "health"},
        {"text": "Political opinion piece on recent elections.", "category": "politics"},
        {"text": "Music review of a new album release.", "category": "entertainment"},
        {"text": "Investment guide for beginners.", "category": "finance"},
        {"text": "Travel diary of a backpacker in Asia.", "category": "travel"},
        {"text": "Medical case study on diabetes treatment.", "category": "medical"},
        {"text": "Technology trends report for the year.", "category": "technology"},
        {"text": "Geography textbook on world continents.", "category": "education"},
        {"text": "Legal contract for business partnership.", "category": "legal"},
        {"text": "Sports interview with an athlete.", "category": "sports"},
        {"text": "Environmental policy document.", "category": "environment"},
        {"text": "Fashion magazine feature on haute couture.", "category": "fashion"},
        {"text": "Guidelines and Process flow for the event.Registeration and Team Formation. ",
            "category": "education"},
        {"text": "Healthy eating tips and recipes.", "category": "health"}
        # Add more documents and categories as needed
    ]
    return dataset
    # Your existing code for creating a synthetic dataset

# Main function for document classification


def classify_document(synthetic_dataset, pdf_text):
    pdf_text = extract_text_from_pdf(pdf_path)
    preprocessed_pdf_text = preprocess_text(pdf_text)

    # Create synthetic labeled dataset
    X = [entry["text"] for entry in synthetic_dataset]
    y = [entry["category"] for entry in synthetic_dataset]

    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)
    pdf_text_vec = vectorizer.transform([preprocessed_pdf_text])

    # Train a Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_vec, y)

    # Make predictions on the preprocessed PDF text
    predicted_category = classifier.predict(pdf_text_vec)

    return predicted_category[0]
    # Your existing code for document classification

# Define an endpoint for document classification


@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Get the PDF file from the request
        pdf_file = request.files['pdf']

        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(pdf_file)

        # Preprocess the text
        preprocessed_pdf_text = preprocess_text(pdf_text)

        # Create a synthetic labeled dataset
        synthetic_dataset = create_labeled_dataset()

        # Classify the document
        predicted_category = classify_document(
            synthetic_dataset, preprocessed_pdf_text)

        return jsonify({'predicted_category': predicted_category})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
