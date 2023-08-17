import requests
import openpyxl
from bs4 import BeautifulSoup
import pandas as pd

"""
def extract_hyperlinks_from_excel(file_path, sheet_name):
    try:
        wb = openpyxl.load_workbook(file_path)
        sheet = wb[sheet_name]

        hyperlinks = []
        for row in sheet.iter_rows():
            for cell in row:
                if cell.hyperlink:
                    # Extract hyperlink URL and display text
                    hyperlink = cell.hyperlink.target
                    hyperlinks.append(hyperlink)

        return hyperlinks
    except Exception as e:
        print(f"Error while reading hyperlinks from Excel: {e}")
        return None


# Replace 'your_file.xlsx' with the actual file path and 'Sheet1' with the sheet name if it's different
file_path = 'ISCO-88&68_index.xlsx'
sheet_name = 'Sheet1'
urls = extract_hyperlinks_from_excel(file_path, sheet_name)

def crawl_website(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract text from elements with class="title"
            title_elements = soup.find_all(class_='title')
            title_text = ' '.join(element.get_text() for element in title_elements)

            # Extract text from elements with class="cb"
            cb_elements = soup.find_all(class_='cb')
            cb_text = ' '.join(element.get_text() for element in cb_elements)

            # Combine the extracted text from both classes
            all_text = title_text + ' ' + cb_text

            return all_text
        else:
            print(f"Failed to crawl {url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error while crawling {url}: {e}")
        return None


with open("isco88index.txt", 'w') as file:
    collected_text = []
    tmp = " "
    for i, url in enumerate(urls):
        if url != tmp:
            text = crawl_website(url)
            if text:
                collected_text.append(text)
                file.write(text)
            tmp = url

naf_data = pd.read_pickle('NAF_index')
naf_text = naf_data['code']+' '+naf_data['desc']+' '+naf_data['desc_long']+' '+naf_data['desc_short']
naf_text = ' '.join(naf_text.tolist())
naf_text = naf_text.replace('\n','')
with open("nafindex.txt", 'w') as file:
    file.write(naf_text)
"""

pcs_data = pd.read_excel('PCS_index.xlsx')
label = ''
pcs_data['text'] = pcs_data['code_pcs']+pcs_data['profession_txt']
pcs_text = ' '.join(pcs_data['text'].tolist())
pcs_text = pcs_text.replace('\n','')

with open('pcsindex.txt', 'w') as file:
    file.write(pcs_text)