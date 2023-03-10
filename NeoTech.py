"""Agente extractor de datos de artículos
Este agente se encarga de la extracción de datos de la revista Pattern
Analysis and Machine Intelligence. 
Para funcionar, el script requiere que estén instalada la
biblioteca `selenium`. """

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import re
from datetime import datetime
import pandas as pd
from selenium.webdriver.common.proxy import *
import time

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


def extract (n, fecha, numEnlace):
    patron = 'pp. C.-C.'
    i = 0
    if (fecha == None):
        fecha = datetime.now()
    driver = webdriver.Chrome(options=options, chrome_options=chrome_options)
    month = fecha.strftime("%m")
    year = fecha.strftime("%Y")
    driver.get("https://www.computer.org/csdl/journal/tp/"+year+"/"+month)
    time.sleep(2)
    titulo = driver.find_elements(By.CLASS_NAME, "article-title")
    length = len(titulo)
    print("Numero articulos = " + str(length))
    for item in titulo:
        enlaces = item.get_attribute("href")
        driver = webdriver.Chrome()
        driver.get(enlaces)
        titulo = driver.find_element(By.CLASS_NAME, 'article-title')
        metadatos = driver.find_element(By.CLASS_NAME, 'article-metadata')
        keywords = driver.find_element(By.TAG_NAME, 'csdl-article-keywords')
        keyString = keywords.text
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
            numEnlace += 1
            print("\nTITULO: ", titulo.text)
            print("\nFECHA: ", date_object.strftime(date_format))
            print("\nABSTRACT: ", abstract.text)
            print("\nKEYWORDS: ", keyList[1::]) 
            i+=1
            driver.quit()
        if(i==length): 
            new_date = pd.to_datetime(fecha)+pd.DateOffset(months=1)
            print("newdate: ", str(new_date))
            extract(n-i, new_date, numEnlace)
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


extract(16, datetime(2006, 2, 1), 1) #Hay 14 en febrero, asi que coge 2 de marzo
extract(2, None, 1)


