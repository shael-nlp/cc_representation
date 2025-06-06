from pypdf import PdfReader

reader = PdfReader("IPCC_AR6_WGIII_SummaryForPolicymakers.pdf")
number_of_pages = len(reader.pages)

all_text = ""
for page_num in range(number_of_pages):
    page = reader.pages[page_num]
    text = page.extract_text()
    if text:
        all_text += text + " "

with open("AR6_WG3_SPM.txt", "w", encoding="utf-8") as file:
    file.write(all_text)