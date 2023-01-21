# -*- coding: utf-8 -*-
"""
@author: morfoula
"""

"""
A standalone script to download and parse edgar 10k MDA section
"""
from sec_api import ExtractorApi

import pandas as pd
extractorApi = ExtractorApi("db04d362d99326e9917b3d3a3abf6afe9f21ef55f5bbf255612ec0bfc4d986b9")

import os


filepath="C:\\Users\\morfo\\Desktop\\10K-MDA-Section-master"   



data = pd.read_excel("Dataset Links.xlsx")

data["Txt File"]

def extract_items_10k(filing_url):
    section_text = extractorApi.get_section(filing_url, "7", "text")

MDA_Dataset=pd.DataFrame(data)
MDA_Dataset=MDA_Dataset.dropna()
MDA_Dataset=MDA_Dataset.iloc[0:352,:]


section_text={'Index': [], 'Data' : []}

for i in range(352):
    f=MDA_Dataset.iloc[i,5]
    section_text['Index'].append(MDA_Dataset.iloc[i,0])
    section_text['Data'].append(extractorApi.get_section(f, "7", "text"))
    
  
section_text=pd.DataFrame(section_text)


final=pd.merge(MDA_Dataset,section_text, on=['Index'])
final.to_excel(r'C:\Users\morfo\Desktop\10K-MDA-Section-master\Dataset.xlsx',index=False)