# This script generates the CSV files in this folder that contain all the elemental decompositions 
# (they are manually edited afterwards to remove comments and irregularities)

# if __name__ == "__main__":
#     import requests
#     from bs4 import BeautifulSoup
#     import os

#     script_dir = os.path.dirname(os.path.abspath(__file__))

#     base_url = 'https://physics.nist.gov/PhysRefData/XrayMassCoef/'
#     table3_url = base_url + 'tab3.html'
#     element_url_template = base_url + 'ElemTab/z{z:02d}.html'

#     response = requests.get(table3_url)
#     soup = BeautifulSoup(response.text, 'html.parser')

#     elements = []
#     for row in soup.find_all('tr'):
#         if "ElemTab" not in str(row):
#             continue
#         cols = [td for td in row.find_all('td') if not td.has_attr('rowspan')]
#         for offset in range(0, len(cols), 3):
#             atomic_number = cols[0 + offset].text.strip()
#             symbol = cols[1 + offset].text.strip()
#             link = cols[2 + offset].find('a')['href']
#             name = cols[2 + offset].find('a').text.strip()
#             elements.append((atomic_number, symbol, name))

#     for z, symbol, name in elements:
#         element_url = element_url_template.format(z=int(z))
#         response = requests.get(element_url)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         pre_tags = soup.find_all('pre')
#         ascii_data = ''
#         for pre in pre_tags:
#             if 'Energy' in pre.get_text():
#                 ascii_data = pre.get_text()
#                 break

#         if ascii_data:
#             lines = ascii_data.splitlines()
#             # Detect leading spaces on line 6
#             testing_line = 6
#             if name in ("Ruthenium", "Zinc", "Silicon"): # special cases where ascii table is formatted differently. WHY?
#                 testing_line = 7 
#             leading_spaces = len(lines[testing_line]) - len(lines[testing_line].lstrip())
#             clean_lines = ['Energy  μ/ρ  μen/ρ']
#             for line in lines[testing_line:]:
#                 clean_lines.append(line[leading_spaces:])
#             output = '\n'.join(clean_lines)
#             filename = f"{z}_{symbol}_{name}"
#             with open(os.path.join(script_dir, filename), 'w') as f:
#                 f.write(output)
#             print(f"Saved: {filename}")
#         else:
#             print(f"ASCII data not found for {name}")
