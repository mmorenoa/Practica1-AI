from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import re
from datetime import datetime
import numpy as np
import pandas as pd
import requests

# QUE HACER:
# METER PROXYS
# Array de keyword
# Arreglar lo de mostrar la fecha

#proxies = {'http': 'http://ip:puerto','https': 'https://ip:puerto'}
#session = requests.Session()
#session.proxies.update(proxies)
#session.get('https://httpbin.org/anything')

# para que no carge imagenes
options = Options()
options.headless = True
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--blink-settings=imagesEnabled=false')


def extract (n, fecha, aux):
    driver = webdriver.Chrome(options=options, chrome_options=chrome_options)
    month = fecha.strftime("%m")
    year = fecha.strftime("%Y")
    driver.get("https://www.computer.org/csdl/journal/tp/"+year+"/"+month)
    titulo = driver.find_elements(By.CLASS_NAME, "article-title")
    length = len(titulo)
    print("Numero articulos = " + str(length))
    i = 0
    keywords = []
    for item in titulo:
        enlaces = item.get_attribute("href")
        driver = webdriver.Chrome()
        driver.get(enlaces)
        patron = '.* ....,'
        patron2 = 'pp. C.-C.'
        titulo = driver.find_element(By.CLASS_NAME, 'article-title')
        metadatos = driver.find_element(By.CLASS_NAME, 'article-metadata')
        keywords = driver.find_element(By.TAG_NAME, 'csdl-article-keywords')
        abstract = driver.find_element(By.TAG_NAME, 'article')
        m1 = re.search(patron, metadatos.text)
        dates = m1.group(0) 
        date_format = "%Y%m"
        date_object = datetime.strptime(dates, "%B %Y,") # ARREGLAR LA FECHA
        m2 = re.search(patron2, metadatos.text) # Esto es para que solo salgan articulos y no covers
        if (not m2):
            print("\n----------")
            print(aux, " enlace: ", enlaces)
            print("----------")
            aux += 1
            print("\nTITULO: ", titulo.text)
            print("\nFECHA: ", date_object.strftime(date_format))
            print("\nABSTRACT: ", abstract.text)
            print("\nKEYWORDS: ", keywords.text) # Esto necesita ser []
            i+=1
            print("i:", i)
            driver.quit()
        if(i==length): 
            new_date = pd.to_datetime(fecha)+pd.DateOffset(months=1)
            print("newdate: ", str(new_date))
            extract(n-i, new_date, aux)
        elif i==(n): break
    

extract(2, datetime(2022, 3, 1), 0)





