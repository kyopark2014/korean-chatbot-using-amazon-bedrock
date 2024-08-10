# PDF에서 텍스트, 이미지, 테이블 정보를 추출하기

[pdf-extraction.ipynb](./docs/pdf-extraction.ipynb)에서는 PDF에서 텍스트, 이미지, 표를 추출하는 방법을 설명하고 있습니다.

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

## 텍스트 추출

페이지 단위로 pdf의 text를 extract_text()로 읽은 후에 contents를 생성하여 활용합니다. 

```python
texts = []

for i, page in enumerate(reader.pages):
    texts.append(page.extract_text())
    
    contents = '\n'.join(texts)    
```

## 이미지 추출

### 개별 이미지 추출하기

pdf 파일에 포함된 모든 이미지들을 추출합니다. 

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

페이지 단위로 이미지를 처리하고자 할때 활용합니다. 상세한 내용은 아래를 참조합니다. fitz를 위해 PyMuPDF를 설치하여야 합니다.

- 아래에서는 한 페이지에 이미지가 4개 이상 있는 경우에 page 단위로 이미지를 저장하고 있습니다.
- 간혹 pdf에 이미지가 있음에도 이미지 object 정보가 없는 경우와 이미지가 1개 이상인데 가로 또는 세로가 100픽셀 이상인 경우에 파일로 저장합니다.
- 이미지 처리의 편의를 위해서 dpi는 200으로 설정하였습니다.
- S3에 파일의 내용을 확인시 png파일이 jpg보다 편리합니다. 따라서 파일 저장 포맷을 png로 하고 있습니다. 

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





## 테이블 추출

### PDF에서 테이블 추출의 어려움

[prompt.pdf](./docs/prompt.pdf)와 같이 PDF안에 Header와 Footer가 있을 수 있고, 표과 2개의 페이지에 걸쳐서 있다면, 하나의 표를 만들기 위해, Header/Footer의 제거 및 2개로 나누어진 표를 하나로 합치는 과정이 필요합니다. Header와 Footer의 크기는 문서마다 다를수 있고 같은 문서라도 페이지마다 다를 수 있어서 문서마다 customize가 필요하며 나누어진 표라는것을 인식하는것도 문서마다 다를 수 있어서 표준화된 문서에 대해서만 테이블 결합이 가능합니다. 

![image](https://github.com/user-attachments/assets/d27e4044-b32f-4f36-bc9a-7e1b40056f4d)

아래는 테이블 결합을 한 예입니다. 다만 이것은 다양한 포맷을 가지고 있는 문서들에는 적용이 어렵고, 표준화된 문서에 대해서만 적용 가능합니다. 

|Input (LLM + tool for SQL to DB)|Output (with SQL tool) – Right answer|
|---|---|
|Calculate the price ratio for stock &#x27;ABC&#x27; between 2023&amp;#45; 01&amp;#45;03 and 2023&amp;#45;01&amp;#45;04?|&gt; Entering new AgentExecutor chain... I will need historical stock price data for the two dates Action: Stock DB Action Input: Price of ABC stock on 2023&amp;#45;01&amp;#45;03 and 2023&amp;#45;01&amp;#45;04 &gt; Entering new SQLDatabaseChain chain...Price of ABC stock on 2023&amp;#45;01&amp;#45;03 and 2023&amp;#45;01&amp;#45;04 SQLQuery:SELECT price FROM stocks WHERE stock_ticker = &quot;ABC&quot; AND date BETWEEN &quot;2023&amp;#45; 01&amp;#45;03&quot; AND &quot;2023&amp;#45;01&amp;#45;04&quot; SQLResult: [(232.0,), (225.0,)] Answer: The price of ABC stock on January 03, 2023 was 232.0 and on January 04, 2023 was 225.0. &gt; Finished chain. Observation: The price of ABC stock on January 03, 2023 was 232.0 and on January 04, 2023 was 225.0. Thought: Now I can compute the price ratio Final Answer: The price ratio for stock &#x27;ABC&#x27; between 2023&amp;#45; 01&amp;#45;03 and 2023&amp;#45;01&amp;#45;04 is 232.0/225.0 = 1.0311|
|---|---|



### MarkDown 형식

fitz로 추출한 페이지에서 아래와 같이 find_tables()로 테이블 객체를 찾아서 to_markdown()로 markdown 형태로 추출할 수 있습니다. 

```python
table_md = []
for i, page in enumerate(pages):
    tab = page.find_tables()
    if tab.tables:
        table_md.append(tab[0].to_markdown())
```

추출된 markdown 형태의 table은 RAG에 문서로 등록할 수 있습니다. 

### 이미지 형식

표에 그림이 포함되어 있거나 표의 요약을 RAG에 등록함으로써 RAG 검색의 정확도를 높일수 있습니다. 또한 추출된 이미지는 표에 대한 링크를 생성할 때 활용됩니다. 

```python
tables = []
for i, page in enumerate(pages):
    page_tables = page.find_tables()
    
    if page_tables.tables:
        tab = page_tables[0]
        
        print(tab.to_markdown())
    
        print(f"index: {i}")
        print(f"bounding box: {tab.bbox}")  # bounding box of the full table
        print(f"top-left cell: {tab.cells[0]}")  # top-left cell
        print(f"bottom-right cell: {tab.cells[-1]}")  # bottom-right cell
        print(f"row count: {tab.row_count}, column count: {tab.col_count}") # row and column counts
        print("\n\n")
        
        extract_table_image(page, i, tab.bbox)
```

이때, 이미지 추출을 위한 함수는 아래와 같습니다.

```python
from PIL import Image

def extract_table_image(page, index, bbox):
    pixmap_ori = page.get_pixmap()
    print(f"width: {pixmap_ori.width}, height: {pixmap_ori.height}")
        
    pixmap = page.get_pixmap(dpi=200)  # dpi=300
    #pixels = pixmap.tobytes() # output: jpg
    
    # convert to png
    img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    print(f"width: {pixmap.width}, height: {pixmap.height}")
    
    rate_width = pixmap.width / pixmap_ori.width
    rate_height = pixmap.height / pixmap_ori.height
    print(f"rate_width={rate_width}, rate_height={rate_height}")
    
    crop_img = img.crop((bbox[0]*rate_width, bbox[1]*rate_height, bbox[2]*rate_width, bbox[3]*rate_height))
    
    pixels = BytesIO()
    crop_img.save(pixels, format='PNG')
    pixels.seek(0, 0)

    # get path from key
    objectName = (key[key.find(s3_prefix)+len(s3_prefix)+1:len(key)])
    folder = s3_prefix+'/captures/'+objectName+'/'
                                
    fname = 'table_'+key.split('/')[-1].split('.')[0]+f"_{index}"

    response = s3_client.put_object(
    Bucket=s3_bucket,
        Key=folder+fname+'.png',
        ContentType='image/png',
        Metadata = {
            "ext": 'png',
            "page": str(index)
        },
        Body=pixels
    )
                                                        
    files.append(folder+fname+'.png')
```



## MarkDown Output

[How to use Markdown output](https://pymupdf.readthedocs.io/en/latest/recipes-text.html#how-to-extract-text-as-markdown)와 같이 pdf를 markdown output으로 저장할 수 있습니다. 

```python
import pymupdf4llm
from langchain.text_splitter import MarkdownTextSplitter

# Get the MD text
md_text = pymupdf4llm.to_markdown("input.pdf")  # get markdown for all pages

splitter = MarkdownTextSplitter(chunk_size=40, chunk_overlap=0)

splitter.create_documents([md_text])
```




## Reference

[Table Recognition and Extraction With PyMuPDF](https://medium.com/@pymupdf/table-recognition-and-extraction-with-pymupdf-54e54b40b760)

[How can I extract semi structured tables from PDF using pdfplumber](https://stackoverflow.com/questions/56155676/how-do-i-extract-a-table-from-a-pdf-file-using-pymupdf)

[How do I extract a table from a pdf file using pymupdf](https://stackoverflow.com/questions/56155676/how-do-i-extract-a-table-from-a-pdf-file-using-pymupdf)

[How to save a pandas DataFrame table as a png](https://stackoverflow.com/questions/35634238/how-to-save-a-pandas-dataframe-table-as-a-png)

[Preprocessing for complex PDF](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/9782bcc083e72693cc1b26aa81b2201bb3d3b07c/genai/aws-gen-ai-kr/20_applications/02_qa_chatbot/01_preprocess_docs/05_0_load_complex_pdf_kr_opensearch.ipynb)

[PyMuPDF4LLM을 활용한 PDF 파싱 및 FAISS 벡터 스토어를 사용한 RAG](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/genai/aws-gen-ai-kr/10_advanced_question_answering/06_rag_faiss_with_images.ipynb)

[Generate Synthetic QnAs from Real-world Data](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/generate-synthetic-qnas-from-real-world-data-on-azure/ba-p/4202053)

[Generate QnA synthetic dataset from a Complex PDF](https://github.com/Azure/synthetic-qa-generation/blob/main/seed/make_qa_multimodal_pdf_docai.ipynb)

[Introducing PyMuPDF4LLM](https://medium.com/@pymupdf/introducing-pymupdf4llm-d2c39442f445)

[Welcome to PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)

[find_table()](https://pymupdf.readthedocs.io/en/latest/page.html#Page.find_tables)
