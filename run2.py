import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field
import warnings

# Suppress specific warnings related to chunk size
warnings.filterwarnings("ignore", category=UserWarning, message="Created a chunk of size")

# Function to extract and clean address
def clean_address(text):
    return text.strip() if text else ""

# Path for input and output Excel files
input_excel = r"C:\Users\Anandan Suresh\Documents\Roche\Data enrichment\website and id.xlsx"
output_file = r"C:\Users\Anandan Suresh\Documents\Roche\Data enrichment\output_3.xlsx"

# ScraperAPI key
API_KEY = "60dd55e01441c811ac66433978eaf746"

# Function to fetch HTML content using ScraperAPI
def get_html_content(url):
    try:
        response = requests.get(f"http://api.scraperapi.com?api_key={API_KEY}&url={url}")
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to retrieve content from {url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return None

# Load the input Excel file
input_df = pd.read_excel(input_excel)

# Initialize the chat model for extracting non-social media details
model_local = ChatOllama(model="llama3.1:8b", temperature=0)

# Initialize an empty DataFrame to hold the results
columns = ["unique id", "official website", "name of organization", "address", "email_address", "contact number", "facebook link", "twitter link", "instagram link", "linkedin id", "youtube link"]
results_df = pd.DataFrame(columns=columns)

# Social media domains to search for
social_media_domains = {
    'facebook': 'facebook link',
    'twitter': 'twitter link',
    'instagram': 'instagram link',
    'linkedin': 'linkedin id',
    'youtube': 'youtube link'
}

# Function to extract social media links
def extract_social_media_links(soup):
    social_media_links = {v: "" for v in social_media_domains.values()}
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        for domain, col_name in social_media_domains.items():
            if domain in href:
                if social_media_links[col_name]:
                    social_media_links[col_name] += ", " + href
                else:
                    social_media_links[col_name] = href
    return social_media_links

# Function to find and extract the contact page link
def find_contact_page(soup, base_url):
    contact_page_url = None
    for a_tag in soup.find_all('a', href=True):
        if 'contact' in a_tag.get_text().lower():
            contact_page_url = a_tag['href']
            if contact_page_url.startswith('/'):
                contact_page_url = base_url + contact_page_url
            break
    return contact_page_url

# Iterate through each row of the input Excel file
for index, row in input_df.iterrows():
    official_website = row['official_website']
    unique_id = row['unique_id']

    print(f"Processing website: {official_website} with unique ID: {unique_id}")

    # Fetch HTML content using ScraperAPI
    html_content = get_html_content(official_website)
    
    # If content is successfully fetched
    if not html_content:
        print(f"Failed to retrieve or fetch content from {official_website}. Skipping...")
        continue

    # Load and parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Extract social media links using the custom logic
    social_media_links = extract_social_media_links(soup)

    # Find the contact page link
    contact_page_url = find_contact_page(soup, official_website)
    
    if contact_page_url:
        print(f"Contact page found: {contact_page_url}")
        contact_page_content = get_html_content(contact_page_url)
        if contact_page_content:
            contact_soup = BeautifulSoup(contact_page_content, "html.parser")
            # Update the soup to use the content from the contact page
            soup = contact_soup
        else:
            print(f"Failed to retrieve content from contact page: {contact_page_url}")
    else:
        print("No contact page found. Continuing with homepage content.")

    # Save the HTML content into a temporary file for processing with BSHTMLLoader
    temp_html_path = f"temp_{unique_id}.html"
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(str(soup))

    # Load the HTML file using BSHTMLLoader
    loader = BSHTMLLoader(temp_html_path, bs_kwargs={"features": "html.parser"}, open_encoding="utf-8")
    data = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7000, chunk_overlap=500)
    doc_splits = text_splitter.split_documents(data)

    # Create the FAISS database from document splits for retrieving other information
    db = FAISS.from_documents(doc_splits, OllamaEmbeddings(model='nomic-embed-text'))
    retriever = db.as_retriever()

    # Define the output format model with organization name, address, etc.
    class FormatJson(BaseModel):
        organization_name: str = Field(default=None, description="Organization name from the given context")
        address: str = Field(default=None, description="Address from the given context")
        email_address: str = Field(default=None, description="Email address from the given context")
        contact_number: str = Field(default=None, description="Contact number from the given context")

    # Create prompt template for the RAG chain
    after_rag_template = """Answer the question based only on the following context:
        {context} and only provide these details in this order: organization name, address, email address, and contact number in a json format.
        {format_instructions}
        Question: {question}
        """
    parser = JsonOutputParser(pydantic_object=FormatJson)
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template, partial_variables={"format_instructions": parser.get_format_instructions()})

    # Define the RAG chain for extracting organization details
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | parser
    )

    try:
        # Invoke the RAG chain with the query for each document
        raw_output = after_rag_chain.invoke("Provide the organization name, address, email address, and contact number from the context")
        
        # Handle the output and ensure it's in the expected format
        json_output = raw_output if isinstance(raw_output, dict) else {}

        # Convert JSON output to a dictionary
        extracted_data = {
            "name of organization": json_output.get("organization_name", ""),
            "address": clean_address(json_output.get("address", "")),
            "email_address": json_output.get("email_address", ""),
            "contact number": json_output.get("contact_number", "")
        }

    except Exception as e:
        print(f"Error occurred while processing the website {official_website}: {e}")
        extracted_data = {}

    # Combine the organization details with the social media links
    combined_data = {
        "unique id": unique_id,
        "official website": official_website,
        **extracted_data,
        **social_media_links
    }

    # Convert combined_data to a DataFrame
    combined_df = pd.DataFrame([combined_data])

    # Concatenate the result to results_df
    results_df = pd.concat([results_df, combined_df], ignore_index=True)

    # Clean up the temporary HTML file
    os.remove(temp_html_path)

    # Save the DataFrame to an Excel file
    results_df.to_excel(output_file, index=False, engine='openpyxl')

    print(f"Data successfully saved to {output_file}")
