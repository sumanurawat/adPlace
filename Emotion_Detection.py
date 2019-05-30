import requests
import cv2
import operator
import numpy as np
import time
from __future__ import print_function
from google.colab import drive
drive.mount('/content/gdrive')

# Defining the variables for API connection
url = 'API_ENDPOINT'
key = 'API_KEY'
CountRetry = 20
root_path = 'gdrive/My Drive/FaceSentimentGermany/'

def FrameCapture(path):
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1
    
    while success:
        success, image = vidObj.read()
        cv2.imwrite("pATH TO WRITE TO" % count, image)
        print(count)
        count += 1

def processRequest( json, data, headers, params ):
    retries = 0
    result = None
    
    while True:
        
        response = requests.request( 'post', url, json = json, data = data, headers = headers, params = params )
        
        if response.status_code == 429:
            
            print( "Message: %s" % ( response.json()['error']['message'] ) )
            
            if retries <= CountRetry:
                time.sleep(1)
                retries += 1
                continue
            else:
                print( 'Error: failed after retrying!' )
                break

    elif response.status_code == 200 or response.status_code == 201:
        print("Here")
        if 'content-length' in response.headers and int(response.headers['content-length']) == 0:
            result = None
            elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str):
                if 'application/json' in response.headers['content-type'].lower():
                    result = response.json() if response.content else None
                    print(result)
                elif 'image' in response.headers['content-type'].lower():
                    result = response.content
                    print(result)
        else:
            print( "Error code: %d" % ( response.status_code ) )
            print( "Message: %s" % ( response.json()['error']['message'] ) )

break
    
    return result

# Divide the video into frames using openCV
pathToFileInDisk = r'gdrive/My Drive/FaceSentimentGermany/Trailer-1.mp4'
FrameCapture(pathToFileInDisk)

pathToFileInDisk1 = r'gdrive/My Drive/FaceSentimentGermany/TrailerTest/frame13.jpg'
with open( pathToFileInDisk1, 'rb' ) as f:
    data = f.read()
headers = dict()
headers['Ocp-Apim-Subscription-Key'] = key
headers['Content-Type'] = 'application/octet-stream'

json = None
params = {
    'returnFaceAttributes': 'emotion',
}


result = processRequest( json, data, headers, params )
print(result)
