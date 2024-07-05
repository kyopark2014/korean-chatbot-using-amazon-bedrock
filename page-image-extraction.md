# 문서를 페이지 단위로 추출하기

## PDF

텍스트 추출을 위해 pypdf를 설치하고, PyMuPDF로 이미지를 읽어옵니다. PyMuPDF로 읽어올 경우에 한글 문서 인식에 대한 추가 변환이 필요하여 편의상 pypdf를 사용합니다. 

```text
pip install pypdf
pip install PyMuPDF
```

S3에서 PDF 문서를 가져온 후에 문서를 byte contents로 로드한 후 PdfRader를 통해 읽어옵니다. 이미지는 PyMuPDF의 fitz.open()을 이용해 page 단위로 읽어옵니다. 각 Page에 대한 이미지는 get_pixmap()을 이용해 byte 이미지로 변환합니다. 이때 바로 저장하면 jpeg로 저장되는데 편의상 png 파일로 변환합니다. 이후 S3에 저장합니다.

```python
from pypdf import PdfReader  

s3r = boto3.resource("s3")
doc = s3r.Object(s3_bucket, key)

Byte_contents = doc.get()['Body'].read()

try: 
    texts = []
    reader = PdfReader(BytesIO(Byte_contents))

    # extract text from pdf
    for i, page in enumerate(reader.pages):
        texts.append(page.extract_text())

    contents = '\n'.join(texts)

    # extract page images using PyMuPDF
    pages = fitz.open(stream=Byte_contents, filetype='pdf')      

    picture_count = 1
    for i, page in enumerate(pages):
        print('page: ', page)
        
        # save current pdf page to image 
        pixmap = page.get_pixmap(dpi=200)  # dpi=300
        #pixels = pixmap.tobytes() # output: jpg
        
        # convert to png
        img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        pixels = BytesIO()
        img.save(pixels, format='PNG')
        pixels.seek(0, 0)
                        
        # get path from key
        objectName = (key[key.find(s3_prefix)+len(s3_prefix)+1:len(key)])
        folder = s3_prefix+'/captures/'+objectName+'/'
        print('folder: ', folder)
                
        fname = 'img_'+key.split('/')[-1].split('.')[0]+f"_{picture_count}"  
        print('fname: ', fname)          
        picture_count = picture_count+1          

        response = s3_client.put_object(
            Bucket=s3_bucket,
            Key=folder+fname+'.png',
            ContentType='image/png',
            Metadata = {
                "ext": 'png',
                "page": str(i+1)
            },
            Body=pixels
        )
        print('response: ', response)
                                        
        files.append(fname)
                        
    contents = '\n'.join(texts)

except Exception:
    err_msg = traceback.format_exc()
    print('err_msg: ', err_msg)
    # raise Exception ("Not able to load the pdf file")
```
