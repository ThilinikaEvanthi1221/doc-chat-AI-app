from flask import Flask, render_template, request
import PyPDF2
from transformers import pipeline
import os

app = Flask(__name__)

# Load the PDF file and extract text
def load_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "  # Add space between pages
                else:
                    print("No text found on this page.")
        print(f"Extracted text: {text[:500]}")  # Print the first 500 characters of the extracted text for debugging
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

# Load the PDF and models
def load_model():
    try:
        print("Loading question-answering model...")
        # Load the model with your Hugging Face token
        huggingface_token = os.getenv("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")  # Set this in your environment
        qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", use_auth_token=huggingface_token)
        print("Model loaded successfully!")
        return qa_pipeline
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Global variables
pdf_text = load_pdf('cricket.pdf')  # Ensure 'cricket.pdf' is in the same directory
qa_pipeline = load_model() if pdf_text else None

@app.route("/", methods=["GET", "POST"])
def chat():
    answer = "Welcome to the PDF Chatbot!"
    question = ""
    
    if request.method == "POST":
        question = request.form["question"]
        
        # If PDF and model are loaded, process the question
        if pdf_text and qa_pipeline:
            result = qa_pipeline(question=question, context=pdf_text)
            if result:
                answer = result['answer']  # Removed confidence score
            else:
                answer = "I'm sorry, I couldn't find an answer."
        else:
            answer = "The PDF or model is not loaded properly."
    
    return render_template("chat.html", question=question, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
