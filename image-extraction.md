# 문서에서 이미지 추출 

문서의 이미지를 추출하여 Amazon S3에 저장한 후에, Multimodal LLM을 이용해 분석하고자 합니다. 문서에서 추출된 이미지는 S3 저장되는데, 이때 발생한 event를 이용하여 event driven 방식으로 이미지에서 텍스트를 추출하고 요약을 수행합니다. 추출된 텍스트와 요약은 RAG에서 활용합니다. 상세한 코드는 [lambda-document](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/lambda-document-manager/lambda_function.py)를 참조합니다. 

## PPTX

"python-pptx"(pip install python-pptx)를 이용해 slide 단위로 Shape를 읽어옵니다. shape_type이 MSO_SHAPE_TYPE.PICTURE일 경우에 S3에 저장합니다.

```python
prs = Presentation(BytesIO(Byte_contents))

def extract_images_from_ppt(prs, key):
    picture_count = 1
    
    extracted_image_files = []
    for i, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            print('shape type: ', shape.shape_type)
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                image = shape.image
                image_bytes = image.blob
                pixels = BytesIO(image_bytes)
                pixels.seek(0, 0)
                        
                # get path from key
                objectName = (key[key.find(s3_prefix)+len(s3_prefix)+1:len(key)])
                folder = s3_prefix+'/files/'+objectName+'/'
                print('folder: ', folder)
                        
                fname = 'img_'+key.split('/')[-1].split('.')[0]+f"_{picture_count}"  
                print('fname: ', fname)
                        
                img_key = folder+fname+'.png'
                        
                response = s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=img_key,
                    ContentType='image/png',
                    Body=pixels
                )
                print('response: ', response)
                picture_count += 1
                
                extracted_image_files.append(img_key)
    
    return extracted_image_files
```

## PDF

"pypdf"(pip install pypdf)를 이용해, 이미지 파일을 추출후 S3에 저장합니다.

```python
from pypdf import PdfReader

s3r = boto3.resource("s3")
doc = s3r.Object(s3_bucket, key)

Byte_contents = doc.get()['Body'].read()

reader = PdfReader(BytesIO(Byte_contents))
            
if enableImageExtraction == 'true':
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
            print('contentType: ', contentType)
                
            if contentType:                
                image_bytes = image_file_object.data
    
                pixels = BytesIO(image_bytes)
                pixels.seek(0, 0)
                                
                objectName = (key[key.find(s3_prefix)+len(s3_prefix)+1:len(key)])
                folder = s3_prefix+'/files/'+objectName+'/'
                img_key = folder+img_name
                    
                response = s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=img_key,
                    ContentType=contentType,
                    Body=pixels
                )
                                
                picture_count += 1                        
                extracted_image_files.append(img_key)
        
    print('extracted_image_files: ', extracted_image_files)    
    return extracted_image_files
```        

## DOCX

"python-docx"(pip install python-docx)를 이용해 이미지 파일을 추출후 S3에 저장합니다.

```python
s3r = boto3.resource("s3")
doc = s3r.Object(s3_bucket, key)

Byte_contents = doc.get()['Body'].read()
doc_contents =docx.Document(BytesIO(Byte_contents))

if enableImageExtraction == 'true':
    image_files = extract_images_from_docx(doc_contents, key)                  
    for img in image_files:
        files.append(img)        

def extract_images_from_docx(doc_contents, key):
    picture_count = 1
    extracted_image_files = []
        
    for inline_shape in doc_contents.inline_shapes:
        if inline_shape.type == WD_INLINE_SHAPE_TYPE.PICTURE:
            rId = inline_shape._inline.graphic.graphicData.pic.blipFill.blip.embed            
            image_part = doc_contents.part.related_parts[rId]            
            filename = image_part.filename            
            bytes_of_image = image_part.image.blob
            pixels = BytesIO(bytes_of_image)
            pixels.seek(0, 0)
                        
            objectName = (key[key.find(s3_prefix)+len(s3_prefix)+1:len(key)])
            folder = s3_prefix+'/files/'+objectName+'/'
            fname = 'img_'+key.split('/')[-1].split('.')[0]+f"_{picture_count}"  
                                
            ext = filename.split('.')[-1]            
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
                        
            img_key = folder+fname+'.'+ext                
            response = s3_client.put_object(
                Bucket=s3_bucket,
                Key=img_key,
                ContentType=contentType,
                Body=pixels
            )                                                                
            picture_count += 1                        
            extracted_image_files.append(img_key)
    print('extracted_image_files: ', extracted_image_files)    
    return extracted_image_files        
```

## 이미지 분석

텍스트의 추출은 아래와 같이 수행합니다. 

```python
def extract_text(chat, img_base64):    
    query = "텍스트를 추출해서 utf8로 변환하세요. <result> tag를 붙여주세요."
    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = chat.invoke(messages)
        
        extracted_text = result.content
        # print('result of text extraction from an image: ', extracted_text)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return extracted_text
```

이미지를 아래와 같이 요약합니다.

```python
def summary_image(chat, img_base64):    
    query = "이미지가 의미하는 내용을 풀어서 자세히 알려주세요. <result> tag를 붙여주세요."
    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = chat.invoke(messages)
        
        extracted_text = result.content
        # print('summary from an image: ', extracted_text)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return extracted_text
```

### 추출 결과 

문서에서 추출된 이미지는 아래와 같습니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/54a6532b-dc65-47ca-8c2e-8273a5c1d541)

이미지를 요약한 결과는 아래와 같습니다. 

```text
이 이미지는 Kinesis Data Streams에서 실시간 데이터를 처리하고 정적 참조 데이터와 결합하는 아키텍처를 보여줍니다.
IoT 디바이스(Simulated)에서 생성된 실시간 데이터 스트림(my_health_metric_stream)이 Kinesis Data Streams로 전송됩니다.
이 데이터 스트림은 20초 마이크로 배치 윈도우로 처리되며, 디바이스 이름을 기준으로 참조 데이터(reference_patient)와 조인됩니다.
참조 데이터에는 환자 이름, 디바이스 이름, 나이, 비상 연락처 정보가 포함되어 있습니다.
조인된 데이터는 최종적으로 At-Risk Patient Data라는 파티션된 데이터 스토어에 저장되며, 여기에는 연도, 월, 일, 시간, 분 정보가 포함됩니다.
이 프로세스는 my-iot-data-processor라는 작업에 의해 처리됩니다.
```


