import pandas as pd
import requests
import os
from urllib.parse import urljoin

df = pd.read_csv('C:/Users/ytl/Desktop/try.csv')

base_url = 'https://arxiv.org/pdf/'

save_path = 'C:/Users/ytl/Desktop/pdf'
os.makedirs(save_path, exist_ok=True)

for paper_id in df['id']:
    pdf_url = urljoin(base_url, f"{paper_id}.pdf")

    pdf_filename = os.path.join(save_path, f"{paper_id}.pdf")

    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        with open(pdf_filename, 'wb') as f:
            f.write(response.content)
    except requests.RequestException as e:
        print(f"Failed to download {paper_id}.pdf: {e}")

print("Download process completed.")
