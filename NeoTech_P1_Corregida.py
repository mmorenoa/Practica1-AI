"""Agente extractor de datos de artículos
Este agente se encarga de la extracción de datos de la revista Pattern
Analysis and Machine Intelligence. 
Para funcionar, el script requiere que estén instalada la
biblioteca `selenium`"""

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import re
from datetime import datetime
import pandas as pd
from selenium.webdriver.common.proxy import *
import time
from dateutil.relativedelta import relativedelta

options = Options()
options.headless = True
chrome_options = webdriver.Chrome(ChromeDriverManager().install()).create_options()
chrome_options.add_argument('--blink-settings=imagesEnabled=false')
myProxy = "185.238.228.67:80"
proxy = Proxy({
    'proxyType': ProxyType.MANUAL,
    'httpProxy': myProxy,
    'sslProxy': myProxy,
    'noProxy': ''})
options.proxy = proxy

result = []
def extract (n, since = None, numEnlace=0):
    if (since == None):
        fecha = datetime.now()
    else: fecha = since
    patron = 'pp. C.-C.'
    i = 0
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options, chrome_options=chrome_options)
    month = fecha.strftime("%m")
    year = fecha.strftime("%Y")
    driver.get("https://www.computer.org/csdl/journal/tp/"+year+"/"+month)
    time.sleep(2)
    titulo = driver.find_elements(By.CLASS_NAME, "article-title")
    tituloLength = len(titulo)
    print("Numero articulos = " + str(tituloLength))
    for item in titulo:
        enlaces = item.get_attribute("href")
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=options, chrome_options=chrome_options)
        driver.get(enlaces)
        time.sleep(2)
        titulo = driver.find_element(By.CLASS_NAME, 'article-title')
        metadatos = driver.find_element(By.CLASS_NAME, 'article-metadata')
        keywords = driver.find_element(By.TAG_NAME, 'csdl-article-keywords')
        keyString = keywords.text
        numEnlace += 1
        keyList = keyString.split(",")
        abstract = driver.find_element(By.TAG_NAME, 'article')
        date_object = formateoFecha(metadatos.text)
        date_format = "%Y%m"
        m3 = re.search(patron, metadatos.text) 
        if (not m3):
            print("\n-----------------------------------------------------------------------------------------\n\n")
            print("\n----------")
            print(numEnlace, " enlace: ", enlaces)
            print("----------")
            print("\nTITULO: ", titulo.text)
            print("\nFECHA: ", date_object.strftime(date_format))
            print("\nABSTRACT: ", abstract.text)
            print("\nKEYWORDS: ", keyList[1::])
            info = (titulo.text, date_object.strftime(date_format), abstract.text, keyList[1::]) 
            result.append(info)
            i+=1
            driver.quit()
        if(i==tituloLength): 
            new_date = pd.to_datetime(fecha)-relativedelta(months=1)
            print("newdate: ", str(new_date))
            extract(n-i, new_date,numEnlace)
        elif i==(n): break
    

def formateoFecha (md):
    patron = '.* ....,'
    m = re.search(patron, md)
    dates = m.group(0) 
    dates = dates.replace("Jan.", "January")
    dates = dates.replace("Feb.", "February")
    dates = dates.replace("Aug.", "August")
    dates = dates.replace("Sept.", "September")
    dates = dates.replace("Oct.", "October")
    dates = dates.replace("Nov.", "November")
    dates = dates.replace("Dec.", "December")
    date_object = datetime.strptime(dates, "%B %Y,") 
    return date_object


#extract(2, datetime(2006, 3, 1),0) 
extract(2)
print("\n\n--------resultado------\n")
print(result)
print("\n")