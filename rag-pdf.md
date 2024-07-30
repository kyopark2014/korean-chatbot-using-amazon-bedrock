# PDF 파일을 이용한 RAG 문서 

## S3로부터 PDF Loading하기

pypdf를 이용하여 S3에 있는 pdf 파일을 로드합니다. 

```python
import boto3
from pypdf import PdfReader      
from io import BytesIO

s3r = boto3.resource("s3")
doc = s3r.Object(s3_bucket, key)

Byte_contents = doc.get()['Body'].read()

reader = PdfReader(BytesIO(Byte_contents))
```

## Text

페이지 단위로 pdf의 text를 extract_text()로 읽은 후에 contents를 생성하여 활용합니다. 

```python
texts = []

for i, page in enumerate(reader.pages):
    texts.append(page.extract_text())
    
    contents = '\n'.join(texts)    
```

## Image

### 개별 이미지 추출하기

pdf 파일에 포함된 이미지를 추출합니다. 

```python
files = []

image_files = extract_images_from_pdf(reader, key)
for img in image_files:
    files.append(img)

def extract_images_from_pdf(reader, key):
    picture_count = 1
    
    extracted_image_files = []
    for i, page in enumerate(reader.pages):        
        for image_file_object in page.images:
            img_name = image_file_object.name
            if img_name in extracted_image_files:
                print('skip....')
                continue
            
            extracted_image_files.append(img_name)
            
            ext = img_name.split('.')[-1]            
            contentType = ""
            if ext == 'png':
                contentType = 'image/png'
            elif ext == 'jpg' or ext == 'jpeg':
                contentType = 'image/jpeg'
            elif ext == 'gif':
                contentType = 'image/gif'
            elif ext == 'bmp':
                contentType = 'image/bmp'
            elif ext == 'tiff' or ext == 'tif':
                contentType = 'image/tiff'
            elif ext == 'svg':
                contentType = 'image/svg+xml'
            elif ext == 'webp':
                contentType = 'image/webp'
            elif ext == 'ico':
                contentType = 'image/x-icon'
            elif ext == 'eps':
                contentType = 'image/eps'
            
            if contentType:                
                image_bytes = image_file_object.data

                pixels = BytesIO(image_bytes)
                pixels.seek(0, 0)
                            
                # get path from key
                objectName = (key[key.find(s3_prefix)+len(s3_prefix)+1:len(key)])
                folder = s3_prefix+'/files/'+objectName+'/'
                            
                img_key = folder+img_name                
                response = s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=img_key,
                    ContentType=contentType,
                    Body=pixels
                )
                            
                # metadata
                img_meta = {   
                    'bucket': s3_bucket,
                    'key': img_key,
                    'url': path+img_key,
                    'ext': 'png',
                    'page': i+1,
                    'original': key
                }                            
                picture_count += 1                    
                extracted_image_files.append(img_key)

    return extracted_image_files
```

### 페이지 단위로 이미지 저장하기

페이지 단위로 이미지를 처리하고자 할때 활용합니다. 아래에서는 한 페이지에 이미지가 4개 이상 있는 경우에 page 단위로 이미지를 저장하고 있습니다. 또한 간혹 pdf에 이미지가 있음에도 이미지 object 정보가 없는 경우와 이미지가 1개 이상인데 가로 또는 세로가 100픽셀 이상인 경우에 파일로 저장합니다. 또 이미지 처리의 편의를 위해서 dpi는 200으로 설정하였습니다. S3에 파일의 내용을 확인시 png파일이 jpg보다 편리합니다. 따라서 파일 저장 포맷을 png로 하고 있습니다. 

```python
import fitz

pages = fitz.open(stream=Byte_contents, filetype='pdf')      
            
for i, page in enumerate(pages):
    imgInfo = page.get_image_info()
    width = height = 0
    for j, info in enumerate(imgInfo):
        bbox = info['bbox']
        print(f"page[{i}] -> bbox[{j}]: {bbox}")
        if (bbox[2]-bbox[0]>width or bbox[3]-bbox[1]>height) and (bbox[2]-bbox[0]<940 and bbox[3]-bbox[1]<520):
            width = bbox[2]-bbox[0]
            height = bbox[3]-bbox[1]
                        
    if nImages[i]>=4 or \
        (nImages[i]>=1 and (width==0 and height==0)) or \
        (nImages[i]>=1 and (width>=100 or height>=100)):
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
                                
        fname = 'img_'+key.split('/')[-1].split('.')[0]+f"_{i}"

        response = s3_client.put_object(
        Bucket=s3_bucket,
            Key=folder+fname+'.png',
            ContentType='image/png',
            Metadata = {
                "ext": 'png',
                "page": str(i)
            },
            Body=pixels
        )
                                                        
        files.append(folder+fname+'.png')                                    
```





## Table

