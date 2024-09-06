import requests
import os
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from time import sleep, time
import nltk
import re
import openai

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Define your OpenAI API key
api_key = "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"

# Function to connect to OpenAI API
def connect_to_openai():
    openai.api_key = api_key

# Define a function to remove stop words from text
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def get_google_search_urls(query, num_results):
    base_url = "https://www.google.com/search?q={}&num=1000&start={}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(base_url.format(query, num_results), headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    search_results = soup.find_all('div', class_='tF2Cxc')
    urls = [result.find('a')['href'] for result in search_results]
    # Limit the number of URLs to num_results
    return urls[:num_results]

def scrape_headers_and_text_from_url(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Raise an exception for bad requests
        soup = BeautifulSoup(response.text, "lxml")
        
        headers_and_text = []
        for header_level in range(1, 7):
            headers = soup.find_all(f'h{header_level}')
            for header in headers:
                header_text = header.get_text()
                header_text = re.sub(r'[^a-zA-Z0-9\s]', '', header_text).strip()
                headers_and_text.append({
                    "url": url,
                    "header_level": f"h{header_level}",
                    "header_text": header_text
                })
        
        return headers_and_text
    except (requests.exceptions.RequestException, Exception) as e:
        print(f"Error scrape: {e}")
        return []

def process_url(url):
    headers_and_text = scrape_headers_and_text_from_url(url)
    
    return headers_and_text

def tfidf_text_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx responses
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from the webpage, excluding script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        text_content = " ".join(soup.stripped_strings)
        return text_content
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return ""  # Return an empty string in case of an error

def calculate_tfidf(contents, min_non_stop_words=10):
    # Modify the vectorizer to include unigrams and bigrams (1-gram and 2-grams)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    combined_contents = [' '.join(content) for content in contents]  # Combine the list of strings into a single string

    # Remove stop words from combined_contents
    combined_contents = [remove_stopwords(text) for text in combined_contents]

    # Filter documents with too few non-stop words
    filtered_contents = []
    for text in combined_contents:
        non_stop_words = [word for word in text.split() if word not in stopwords.words('english')]
        if len(non_stop_words) >= min_non_stop_words:
            filtered_contents.append(text)

    if not filtered_contents:
        return None, None

    tfidf_matrix = vectorizer.fit_transform(filtered_contents)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

def get_user_input(query):
    # Dynamic default prompt
    default_prompt = f"Above is a collection of outlines/headings on the topic of '{query}' taken from the top 20 SERP results of competitors. Drawing on these competitors' outlines collection, \n1. Understand first what is the user intention (pains, problems, needs, wants, goals/dreams). \n2. Then devise 8 distinctive headings centered around the theme of 'KEYWORD'. \n3. Each heading should be highly relevant to the topic, SEO-optimized, cater to user intentions, and integrate NLP keywords. \n4.Strive for a mixture of question-based, declarative, and other engaging heading styles. \n5. Don't include subheadings, introductory, and concluding sections. \n6. The objective is to create an article outline that stands out in the competitive SERP environment while also appealing to readers. \n7. Provide two outputs. The first is the user intention info, and the second is the outline you created."
    print("\nDefault Prompt:")
    print(default_prompt)
    # Allow the user to choose to use the default prompt or change it
    use_default_prompt = input("Do you want to use the default prompt? (yes/no): ").strip().lower()
    if use_default_prompt == 'yes':
        prompt = default_prompt
    else:
        prompt = input("Enter the desired prompt: ")

    return prompt

# Function to clean text
def clean_text(text):
    # Remove redundant spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove non-alphanumeric characters (e.g., â€¯Â™)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Strip leading and trailing spaces
    text = text.strip()
    return text

# Function to process the user's input and generate a response
def generate_response(prompt, csv_file_path):
    try:
        # Read CSV data into a DataFrame
        df = pd.read_csv(csv_file_path)

        # Remove rows with NaN values in the specified column
        df = df.dropna(subset=["header_text"])

         # Clean the "header_text" column
        df["header_text"] = df["header_text"].apply(clean_text)

        # Extract the data from the cleaned DataFrame
        data = df["header_text"].tolist()
        
        # Combine cleaned data and user input prompt into a single string
        combined_prompt = "List Header:" + "\n".join(data) + "\n\nPrompt:" + f"\n{prompt}"

        # Generate the response with increased max tokens
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=combined_prompt,
            max_tokens=1000  # Increase max tokens as needed
        )

        return response.choices[0].text
    except Exception as e:
        print(f"Error: {str(e)}")
        return "", ""

def save_response_to_file(response, query):
    folder = "_Output"
    file_folder = f"{query}"
    output_folder = os.path.join(folder, file_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    txt_filename = f"Outline_{query}.txt"
    output_file_txt_path = os.path.join(output_folder, txt_filename)
    
    try:
        with open(output_file_txt_path, 'w', encoding='utf-8') as file:
            file.write(response)  # Write the response string to the file
    except Exception as e:
        print(f"Error saving response to file: {e}")

def main():
    connect_to_openai()
    while True:  # Add a while loop to allow multiple runs
        query = input("Enter a topic (keyword): ")
        urls = get_google_search_urls(query, num_results=20)
        
        # Define the number of concurrent threads (adjust as needed)
        num_threads = 5
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            start_time = time()
            results = list(executor.map(process_url, urls))
            end_time = time()
            print(f"Scraping and processing completed in {end_time - start_time:.2f} seconds")
        
        # Flatten the list of headers and create a DataFrame
        flattened_results = [header for headers in results for header in headers]
        df = pd.DataFrame(flattened_results)

        # Calculate TF-IDF
        documents = []
        for index, url in enumerate(urls, start=1):
            print(f"{index}. Scraping {url}")
            text_content = tfidf_text_from_url(url)
            words = word_tokenize(text_content) # Tokenize the combined text content

            # Remove stop words from words
            words = [word for word in words if word.lower() not in stopwords.words('english')]

            documents.append(words)
            sleep(2)

        tfidf_matrix, feature_names = calculate_tfidf(documents)
        if tfidf_matrix is not None:
            dense_matrix = tfidf_matrix.todense()
            df1 = pd.DataFrame(dense_matrix, columns=feature_names)

            # Menghitung frekuensi muncul
            freqs = df1[df1 > 0].count(axis=0)

            # Menghitung rata-rata skor TF-IDF
            avg_tfidf = round(df1.mean(axis=0), 2)

            result = pd.DataFrame({
                'Word/Phrase': feature_names,
                'Average TF-IDF Score': avg_tfidf,
                'Frequency': freqs
            })

            result.sort_values(by='Average TF-IDF Score', ascending=False, inplace=True)
            # Remove rows with TF-IDF values equal to 0
            result = result[result['Average TF-IDF Score'] != 0]
            # Reset the row indices of both DataFrames
            df = df.reset_index(drop=True)
            result = result.reset_index(drop=True)
            combined_df = pd.concat([df, result], axis=1)
        else:
            combined_df = df  # If tfidf_matrix is None, use the original DataFrame df
        
        # Define the output directory (you can change this as needed)
        folder = "_Output"
        file_folder = f"{query}"
        output_folder = os.path.join(folder, file_folder)
        os.makedirs(output_folder, exist_ok=True)
        
        # Save CSV
        csv_filename = f"{query}_scrape_tf-idf.csv"
        output_file_csv_path = os.path.join(output_folder, csv_filename)
        combined_df.to_csv(output_file_csv_path, index=False)
            
        print(f"\nHasil Scraping & TF-IDF telah disimpan di {output_folder} dengan nama {csv_filename}.")
        
        prompt = get_user_input(query)
        response = generate_response(prompt, output_file_csv_path)
        
        if response:
            save_response_to_file(response, query)
            print(f"\nHasil Outline telah disimpan di {output_folder}")
        else:
            print("\nNo Outline generated.")
        
        # Ask the user whether to do scraping again
        repeat = input("\nDo you want to run the program again? (yes/no): ").strip().lower()
        if repeat != "yes":
            break  # Exit the loop if the user's choice is not 'Y'

if __name__ == "__main__":
    main()