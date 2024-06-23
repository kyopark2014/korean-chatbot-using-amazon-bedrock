# S3 Event를 이용한 RAG 등록

등록이 필요한 Event Type에서 CREATED_PUT, REMOVED_DELETE, CREATED_COMPLETE_MULTIPART_UPLOAD가 있습니다. 여기서 CREATED_COMPLETE_MULTIPART_UPLOAD은 대용량 파일 업로드시 발생하는 event 입니다.

```python
const s3PutEventSource = new lambdaEventSources.S3EventSource(s3Bucket, {
    events: [
      s3.EventType.OBJECT_CREATED_PUT,
      s3.EventType.OBJECT_REMOVED_DELETE,
      s3.EventType.OBJECT_CREATED_COMPLETE_MULTIPART_UPLOAD
    ],
    filters: [
      { prefix: s3_prefix+'/' },
    ]
  });
  lambdaS3eventManager.addEventSource(s3PutEventSource);
```


## 이미지 추출 (PPTX)

"python-pptx"를 설치후에 slide 단위로 Shape를 읽어오입니다. shape_type이 MSO_SHAPE_TYPE.PICTURE일 경우에 S3에 저장합니다.

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
                # image bytes to PIL Image object
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

S3의 저장 event를 이용해 text를 추출하고 summary를 만듧니다.

이미지 요약은 아래와 같이 수행합니다.

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
