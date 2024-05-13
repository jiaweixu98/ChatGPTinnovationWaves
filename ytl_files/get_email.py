import os
import re
import csv
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf = PdfReader(file)
        # Extract text from only the first page
        if len(pdf.pages) > 0:
            first_page_text = pdf.pages[0].extract_text()
            return first_page_text if first_page_text else ""
        return ""

def extract_emails(text):
    email_pattern = r'\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,6}\b'
    potential_emails = re.findall(email_pattern, text)
    valid_emails = [email for email in potential_emails if not any(kw in email for kw in ['abstract', 'Abstract', 'Introduction', 'introduction'])]
    return valid_emails

def extract_institutions(text):
    lines = text.split('\n')
    institution_keywords = ['University', 'Institute', 'College', 'Laboratory', 'Lab', 'School', 'Department', 'Dept', 'Center', 'Centre', 'Research', 'Campus']
    institutions = []

    for line in lines:
        original_line = line.strip()
        potential_institutions = re.split(r';|\band\b|,', original_line)  # Splitting based on common delimiters and 'and'

        for institution in potential_institutions:
            institution = institution.strip()
            if any(keyword in institution for keyword in institution_keywords):
                end_punctuations = ['.', ',', ';', ':']
                line_endings = [institution.index(punct) for punct in end_punctuations if punct in institution]
                if line_endings:
                    institution = institution[:min(line_endings)]
                if institution and institution[0].isdigit():  # Remove leading numbering if present
                    institution = institution[1:].lstrip()
                institutions.append(institution)

    return institutions

pdf_directory = 'C:/Users/ytl/Desktop/pdf'  #输入为存储pdf的文件夹路径
output_csv_path = 'C:/Users/ytl/Desktop/results.csv'  #保存结果的csv文件路径

with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['ID', 'Institutions', 'Emails'])

    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            file_id = os.path.splitext(filename)[0]
            pdf_path = os.path.join(pdf_directory, filename)

            text = extract_text_from_pdf(pdf_path)
            emails = extract_emails(text)
            email_str = ','.join(emails)

            institutions = extract_institutions(text)
            institution_str = ','.join(set(institutions))

            csvwriter.writerow([file_id, institution_str, email_str])
