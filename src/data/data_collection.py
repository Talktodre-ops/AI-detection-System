import os
from pathlib import Path
import logging
import requests
import openai
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

# Define directories
list_of_dirs = [
    "datasets/AI_generated",
    "datasets/human_written",
    "datasets/plagiarism_sources"
]

# Create directories if they don't exist
for directory in list_of_dirs:
    os.makedirs(directory, exist_ok=True)
    logging.info(f"Ensuring directory exists: {directory}")

# Function to generate AI-written text using OpenAI's API
def generate_ai_text(prompt, filename, num_samples=10):
    """
    Generate AI-written text using OpenAI's API and save it to a file.

    Args:
        prompt (str): The prompt for the AI to generate text.
        filename (str): The filename to save the generated text.
        num_samples (int): The number of samples to generate.
    """
    openai.api_key = ""  # Replace with your API Key
    file_path = Path(f"datasets/AI_generated/{filename}")
    
    with open(file_path, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            f.write(response["choices"][0]["message"]["content"] + "\n\n")
    logging.info(f"AI-generated text saved to: {file_path}")

# Function to scrape human-written text from Wikipedia
def scrape_wikipedia(topic, filename):
    """
    Scrape human-written text from Wikipedia and save it to a file.

    Args:
        topic (str): The Wikipedia topic to scrape.
        filename (str): The filename to save the scraped text.
    """
    url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    response = requests.get(url)
    file_path = Path(f"datasets/human_written/{filename}")
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        
        with open(file_path, "w", encoding="utf-8") as f:
            for p in paragraphs:
                f.write(p.text + "\n")
        logging.info(f"Wikipedia content saved to: {file_path}")
    else:
        logging.warning(f"Failed to fetch {url}")

# Function to download academic papers from arXiv
def download_arxiv_papers(query, num_papers=5):
    """
    Download academic papers from arXiv and save their metadata to a file.

    Args:
        query (str): The search query for arXiv papers.
        num_papers (int): The number of papers to download.
    """
    arxiv_api_url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={num_papers}"
    response = requests.get(arxiv_api_url)
    file_path = Path("datasets/plagiarism_sources/arxiv_papers.xml")
    
    if response.status_code == 200:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        logging.info(f"ArXiv papers metadata saved to: {file_path}")
    else:
        logging.warning("Failed to fetch arXiv papers.")

# Collect AI-generated content
generate_ai_text("Write a 500-word essay on climate change.", "climate_change_ai.txt")
generate_ai_text("Explain the theory of evolution in a research-style format.", "evolution_ai.txt")

# Collect human-written Wikipedia articles
scrape_wikipedia("Climate change", "climate_change_wikipedia.txt")
scrape_wikipedia("Evolution", "evolution_wikipedia.txt")

# Collect plagiarism reference data from arXiv
download_arxiv_papers("machine learning", num_papers=10)

logging.info("Data collection complete!")