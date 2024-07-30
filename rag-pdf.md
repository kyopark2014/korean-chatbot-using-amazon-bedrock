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





## Table

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
|LAST UPDATED: JUNE 2024 AMAZON CONFIDENTIAL – INTERNAL USE ONLY Generative AI Compete Battlecard|Col2|
|---|---|
|Layer AWS Applications Amazon Q Business (Leveraging FMs) Amazon Q Developer Amazon Q in QuickSight Amazon Q in Connect Services Foundation Models &amp; providers (FMs, Tools, Safety &amp; Amazon (Titan Multimodal, Titan Ima Security, Features) Text/Multimodal Embeddings) AI21 (Jurassic Mid, Ultra), Anthropic ( Cohere (Command, Embed, Multiling 3 13B, 70B), Mistral AI (Small, Large, AI (SDXL)|OpenAI Microsoft Azure OpenAI Google Cloud Platform ChatGPT Enterprise, ChatGPT Team Microsoft 365 Copilot Gemini for Google Workspace N/A Github Copilot + Azure Copilot. Needs both to match Q Developer. Gemini for Google Cloud, Gemini Cloud Assist, Gemini Code Assist, Gemini in BigQuery, Gemini in Databases, Gemini in Security N/A Copilot for Fabric, Power BI Gemini in Looker N/A Microsoft Digital Contact Center, Microsoft Copilot for Service Agent Assist (Contact Center AI) Foundation Models &amp; providers Foundation Models &amp; providers Foundation Models &amp; providers ge, Titan OpenAI API FMs (Provider: OpenAI) Azure OpenAI Service FMs (Provider: OpenAI) 150 models (on Model Garden) including: GPT (4o, 4 Turbo, 3.5 Turbo), DALL&amp;#45;E, TTS, GPT (4o, 4 Turbo, 3.5 Turbo), DALL&amp;#45;E, TTS, Whisper, Embeddings Gemini 1.5 Flash, Gemini 1.5 Pro, Gemini 1.0 Pro Vision, Imagen Claude family), Whisper, Embeddings 3, Chirp, Embeddings for text, multimodal, Anthropic (Claude ual), Meta (Llama 2, Model As&amp;#45;a&amp;#45;Service FMs &amp; Providers (pay&amp;#45;as&amp;#45;you go only). family), Meta (Llama 2, 3, Code Llama , Llama 2( Quantized), Microsoft (Phi), Meta (Llama 2, 3), Cohere (Command, Command R+, Instruct), Stability Mistral AI (8x7B, 7B), Stability AI (Stable Diffusion family), Embed), Mistral (Small, Large), G42 (Jais), Nixtla (TimeGEN&amp;#45;1) Databricks (Dolly&amp;#45;v2&amp;#45;7b), TII (Falcon&amp;#45;Instruct &amp;#45;PEFT), Gemma,|
|CodeGemma, PaliGemma, HuggingFace Features Features Features Features Fine&amp;#45;tuning, pre&amp;#45;training, Imported models(Preview), Fine&amp;#45;tuning, Chat Completions, Playground, OpenAI Service, Model as a Service(MaaS), Fine&amp;#45;tuning(Preview), Prompt Gallery, Playgrounds (Multimodal, Language, Vision, Bedrock Studio (Preview), Model evaluation, Provisioned Prompt examples, Batch (discounted) Prompt Flow, Model benchmarks, Tracing (Preview), Evaluation( Speech), Serve, Fine&amp;#45;Tune (OOB Supervised – PeFT, LoRA, RLHF), Throughput, Playground, Quota Management, Prompt Preview) Playground Distill, AutoSxS, Rapid Evaluation, Grounding (in Google Search), Catalog, Converse Notebooks, Training, Feature Store, Pipelines, Deployments, Monitoring, Document Processors, Reasoning Engine, Rankers Tools Tools Tools Tools Bedrock console, Bedrock Studio (Preview) Assistants API (Beta): File Search, Code AI Studio, Model Catalog, AI Toolkit, Developer CLI(azd), Copilot Model Garden, Vertex AI Studio, Vertex AI Agent Builder Studio. Interpreter, Function Calling RAG (including low&amp;#45;code, and no&amp;#45;code) RAG (including low&amp;#45;code, and no&amp;#45;code) RAG (including low&amp;#45;code, and no&amp;#45;code) RAG (including low&amp;#45;code, no&amp;#45;code and code&amp;#45;centric) Knowledge Bases (OOB RAG), Agents Assistants API (Beta) Semantic Kernel, Assistants API, Add your data (Preview). OOB RAG (Google Quality Search, Chat, Recommendations), No&amp;#45; code Agents, Extensions, Function Calling Safety: Guardrails, Safety: Moderation, Safety: Content Filters (Preview), Safety: Safety filters Security: Private Endpoint, VPC, Encryption, Data Security: No regional support, No VPC support Security: Private Endpoint, VPC (a.k.a. VNET), Encryption, Data Security: Security controls — Data residency, Customer&amp;#45;managed Residency Residency encryption key (CMEK), Private Service Connect, VPC&amp;#45;SC, Access Transparency (AXT) Infrastructure Provisioned Throughput, SageMaker, Trainium, Reliant on Microsoft Azure infrastructure, but Azure Machine Learning, Provisioned Throughput, GPUs, TPUs, Dynamic Workload Scheduler (AI Hypercomputer), Inferentia, GPUs, UltraClusters, Elastic Fabric Adapter abstracts it, so customers don’t see the NVDIA GPUs (A100, H100), AMD (MI300x), Provisioned Throughput for Anthropic FMs. (FMs Training and Inference) (EFA), EC2 Capacity Blocks, Nitro, Neuron benefits. Dedicated Instance is the name for Azure Maia 100 provisioned throughput offering. (Table current as of 06/13/2024. See the Generative AI Vendor Capability Matrix for the latest info as well as important context.) See incorrect information on this battlecard? Create a SIM request to update here. © 2024, Amazon Web Services, Inc. or its affiliates. All rights reserved. Amazon Confidential and Trademark.||


|Layer|AWS|OpenAI|Microsoft Azure OpenAI|Google Cloud Platform|
|---|---|---|---|---|
|Applications (Leveraging FMs)|Amazon Q Business|ChatGPT Enterprise, ChatGPT Team|Microsoft 365 Copilot|Gemini for Google Workspace|
||Amazon Q Developer|N/A|Github Copilot + Azure Copilot. Needs both to match Q Developer.|Gemini for Google Cloud, Gemini Cloud Assist, Gemini Code Assist, Gemini in BigQuery, Gemini in Databases, Gemini in Security|
||Amazon Q in QuickSight|N/A|Copilot for Fabric, Power BI|Gemini in Looker|
||Amazon Q in Connect|N/A|Microsoft Digital Contact Center, Microsoft Copilot for Service|Agent Assist (Contact Center AI)|
|Services (FMs, Tools, Safety &amp; Security, Features)|Foundation Models &amp; providers Amazon (Titan Multimodal, Titan Image, Titan Text/Multimodal Embeddings) AI21 (Jurassic Mid, Ultra), Anthropic (Claude family), Cohere (Command, Embed, Multilingual), Meta (Llama 2, 3 13B, 70B), Mistral AI (Small, Large, Instruct), Stability AI (SDXL)|Foundation Models &amp; providers OpenAI API FMs (Provider: OpenAI) GPT (4o, 4 Turbo, 3.5 Turbo), DALL&amp;#45;E, TTS, Whisper, Embeddings|Foundation Models &amp; providers Azure OpenAI Service FMs (Provider: OpenAI) GPT (4o, 4 Turbo, 3.5 Turbo), DALL&amp;#45;E, TTS, Whisper, Embeddings Model As&amp;#45;a&amp;#45;Service FMs &amp; Providers (pay&amp;#45;as&amp;#45;you go only). Microsoft (Phi), Meta (Llama 2, 3), Cohere (Command, Command R+, Embed), Mistral (Small, Large), G42 (Jais), Nixtla (TimeGEN&amp;#45;1)|Foundation Models &amp; providers 150 models (on Model Garden) including: Gemini 1.5 Flash, Gemini 1.5 Pro, Gemini 1.0 Pro Vision, Imagen 3, Chirp, Embeddings for text, multimodal, Anthropic (Claude family), Meta (Llama 2, 3, Code Llama , Llama 2( Quantized), Mistral AI (8x7B, 7B), Stability AI (Stable Diffusion family), Databricks (Dolly&amp;#45;v2&amp;#45;7b), TII (Falcon&amp;#45;Instruct &amp;#45;PEFT), Gemma, CodeGemma, PaliGemma, HuggingFace|
||Features Fine&amp;#45;tuning, pre&amp;#45;training, Imported models(Preview), Bedrock Studio (Preview), Model evaluation, Provisioned Throughput, Playground, Quota Management, Prompt Catalog, Converse|Features Fine&amp;#45;tuning, Chat Completions, Playground, Prompt examples, Batch (discounted)|Features OpenAI Service, Model as a Service(MaaS), Fine&amp;#45;tuning(Preview), Prompt Flow, Model benchmarks, Tracing (Preview), Evaluation( Preview) Playground|Features Prompt Gallery, Playgrounds (Multimodal, Language, Vision, Speech), Serve, Fine&amp;#45;Tune (OOB Supervised – PeFT, LoRA, RLHF), Distill, AutoSxS, Rapid Evaluation, Grounding (in Google Search), Notebooks, Training, Feature Store, Pipelines, Deployments, Monitoring, Document Processors, Reasoning Engine, Rankers|
||Tools Bedrock console, Bedrock Studio (Preview)|Tools Assistants API (Beta): File Search, Code Interpreter, Function Calling|Tools AI Studio, Model Catalog, AI Toolkit, Developer CLI(azd), Copilot Studio.|Tools Model Garden, Vertex AI Studio, Vertex AI Agent Builder|
||RAG (including low&amp;#45;code, and no&amp;#45;code) Knowledge Bases (OOB RAG), Agents|RAG (including low&amp;#45;code, and no&amp;#45;code) Assistants API (Beta)|RAG (including low&amp;#45;code, and no&amp;#45;code) Semantic Kernel, Assistants API, Add your data (Preview).|RAG (including low&amp;#45;code, no&amp;#45;code and code&amp;#45;centric) OOB RAG (Google Quality Search, Chat, Recommendations), No&amp;#45; code Agents, Extensions, Function Calling|
||Safety: Guardrails, Security: Private Endpoint, VPC, Encryption, Data Residency|Safety: Moderation, Security: No regional support, No VPC support|Safety: Content Filters (Preview), Security: Private Endpoint, VPC (a.k.a. VNET), Encryption, Data Residency|Safety: Safety filters Security: Security controls — Data residency, Customer&amp;#45;managed encryption key (CMEK), Private Service Connect, VPC&amp;#45;SC, Access Transparency (AXT)|
|Infrastructure (FMs Training and Inference)|Provisioned Throughput, SageMaker, Trainium, Inferentia, GPUs, UltraClusters, Elastic Fabric Adapter (EFA), EC2 Capacity Blocks, Nitro, Neuron|Reliant on Microsoft Azure infrastructure, but abstracts it, so customers don’t see the benefits. Dedicated Instance is the name for provisioned throughput offering.|Azure Machine Learning, Provisioned Throughput, NVDIA GPUs (A100, H100), AMD (MI300x), Azure Maia 100|GPUs, TPUs, Dynamic Workload Scheduler (AI Hypercomputer), Provisioned Throughput for Anthropic FMs.|

## Reference

[PyMuPDF 상세 설명[(https://pymupdf.readthedocs.io/en/latest/)
