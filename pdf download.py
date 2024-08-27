import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Set up the WebDriver (Chrome in this example)
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Run in headless mode to avoid opening a browser window
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Initialize the WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Navigate to the CIS download page
url = "https://downloads.cisecurity.org/#/"
driver.get(url)
time.sleep(5)  # Wait for the page to load

# Get the page source and parse with BeautifulSoup
page_source = driver.page_source
soup = BeautifulSoup(page_source, 'html.parser')

# Find all product links (assuming they are <a> tags with a specific class or attribute)
product_links = soup.find_all('a', href=True)

# Folder to save the downloaded PDFs
download_folder = "CIS_PDFs"
os.makedirs(download_folder, exist_ok=True)

# Download all PDFs
for link in product_links:
    href = link['href']
    if href.endswith('.pdf'):  # Check if the link is to a PDF file
        pdf_url = href
        pdf_name = os.path.join(download_folder, pdf_url.split('/')[-1])
        
        response = requests.get(pdf_url)
        with open(pdf_name, 'wb') as f:
            f.write(response.content)
        print(f'Downloaded: {pdf_name}')

# Close the WebDriver
driver.quit()
