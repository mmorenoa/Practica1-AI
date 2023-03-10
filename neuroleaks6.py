from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import re
from datetime import datetime
import numpy as np
import pandas as pd
from selenium.webdriver.common.proxy import *

# QUE HACER:
# Array de keyword


# para que no carge imagenes
options = Options()
options.headless = True
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--blink-settings=imagesEnabled=false')

myProxy = "185.238.228.67:80"
proxy = Proxy({
    'proxyType': ProxyType.MANUAL,
    'httpProxy': myProxy,
    'sslProxy': myProxy,
    'noProxy': ''})
options.proxy = proxy



def extract (n, fecha, aux):
    driver = webdriver.Chrome(options=options, chrome_options=chrome_options)
    month = fecha.strftime("%m")
    year = fecha.strftime("%Y")
    driver.get("https://www.computer.org/csdl/journal/tp/"+year+"/"+month)
    titulo = driver.find_elements(By.CLASS_NAME, "article-title")
    length = len(titulo)
    print("Numero articulos = " + str(length))
    i = 0
    key = []
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
        patron3 = "\."
        m2 = re.search(patron3, dates)
        if (m2):
            date_object = datetime.strptime(dates, "%B. %Y,") 
        else: 
            date_object = datetime.strptime(dates, "%B %Y,") 
        m3 = re.search(patron2, metadatos.text) # Esto es para que solo salgan articulos y no covers
        if (not m3):
            print("\n----------")
            print(aux, " enlace: ", enlaces)
            print("----------")
            aux += 1
            print("\nTITULO: ", titulo.text)
            print("\nFECHA: ", date_object.strftime(date_format))
            print("\nABSTRACT: ", abstract.text)
            print("\nKEYWORDS: ", key) 
            i+=1
            print("i:", i)
            driver.quit()
        if(i==length): 
            new_date = pd.to_datetime(fecha)+pd.DateOffset(months=1)
            print("newdate: ", str(new_date))
            extract(n-i, new_date, aux)
        elif i==(n): break
    

extract(1, datetime(1999, 3, 1), 0)


