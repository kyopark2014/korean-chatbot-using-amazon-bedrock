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

### MarkDown으로 변환

|LAST UPDATED: JUNE 2024 AMAZON CONFIDENTIAL – INTERNAL USE ONLY Generative AI Compete Battlecard|Col2|
|---|---|
|Overview AWS has built a comprehensive generative AI stack across three key layers: infrastructure, tools, and applications. Infrastructure includes NVIDIA GPUs and custom silicon, such as Trainium and Inferentia, for Foundation Model (FM) training and inference. Tools, such as Amazon Bedrock, help customers build with FMs. And applications such as Amazon Q leverage these FMs. Microsoft, Google Cloud, and OpenAI all position their own services across each layer of the stack. It’s important to articulate AWS’s differentiators at each layer in order to help customers make informed decisions as they leverage generative AI to transform their businesses. See the Generative AI Compete hub for the latest competitive resources and info.|Key takeaways Additional compete resources Call out AWS value propositions (see Slide 2 for more detail): Bedrock Compete Battlecards: Bedrock &amp; Azure OpenAI Service; 1. Broad selection of FMs from leading providers (e.g., Anthropic, Cohere, Meta, Mistral AI, Stability AI, Amazon) available Bedrock &amp; GCP Vertex AI; Agents for Bedrock through a single API. Amazon Q Compete Battlecards: 2. Data as your differentiator: Customize your FMs using data processed and secured in the best cloud, AWS. Q Business; Q Developer; Q in QuickSight 3. Most price&amp;#45;performant infrastructure for GenAI: broadest set of GPUs &amp; accelerators to train models and run inference. CI Generative AI Compete Hub: Hub for compete&amp;#45;specific GenAI 4. AI&amp;#45;powered applications to boost productivity, such as Amazon Q. resources 5. Enterprise&amp;#45;grade security, privacy, and safety built&amp;#45;in: Bedrock and Amazon Q are built to be secure (e.g. Guardrails). Generative AI Competitive Landscape deck: Understand who we’re Work backward from the customers’ needs: Be prepared to show what is possible with transformative use cases. competing against, what they offer customers, and how we respond Pitch the customer on a Proof of Value exercise. (See “Free trial” section on Q Pricing page.) Generative AI Vendor Capability Matrix: Comparison of products and Pitch AWS as the best company to partner with today and in the long&amp;#45;term: AWS has the most comprehensive cloud use cases across OpenAI, Microsoft Azure, GCP, and AWS offerings, we have the best resources and programs to help them on&amp;#45;board and execute today. Long&amp;#45;term, the models will #generative&amp;#45;ai&amp;#45;compete Slack Channel: Share insights and ask continue to improve &amp; become commodities. questions about generative AI compete with experts from across AWS Highlight AWS’ key role in democratizing AI and ML: AWS and Amazon have a 25+ year history of working on AI/ML, and have helped make ML available to more than 100,000 customers.|
|Competitor overview – Microsoft Azure Competitor overview – OpenAI Competitor overview – Google Cloud Microsoft has infused generative AI across many of their products, including Azure, M365, Dynamics OpenAI is an AI research and deployment Google Cloud leads with a strong message around its natively multimodal Gemini family of models, 365, Windows, and GitHub. They use these to showcase what is possible and gain C&amp;#45;suite mindshare company. Their offerings include: featuring the industry’s longest context window (2 million+ tokens), no&amp;#45;code agents, and enterprise&amp;#45; as a &quot;leader in AI.&quot; They also use generative AI and their exclusive access to OpenAI models as an readiness features. Google entices customers with the infusion of generative AI across its portfolio – ChatGPT (Team and Enterprise) at the enticement to get customers to try Azure. of cloud services. In May 2024, they showcased their future AI strategy with Project Astra, which can application layer. ChatGPT is the most popular AI process multimodal information in real time by understanding the context where the user is present. Their offerings across the generative AI stack include: chatbot; it competes with Q Business. – Copilots at the application layer. Microsoft 365 Copilot and GitHub Copilot are the most Differentiate by emphasizing our ability to use GCP offers generative AI solutions at all 3 layers of the AI/ML stack. Its value proposition includes: prominent, competing with our Q Business and Q Developer offerings. Microsoft has incumbency with customer’s data via our Q Business connectors. – Application layer offerings of AI&amp;#45;enabled assistants: Gemini for Google Workspace (think Amazon these audiences. See Q Business compete battlecard. Q Business) for its productivity suite empowering business users; Gemini for Google Cloud (think We should compete on our value – we provide more features at a lower cost – in these spaces. – The OpenAI API is a REST API that builders can Amazon Q Developer) targeting Cloud users. Check our battle cards for details (Q Business, Q Developer, Q in QuickSight). use to programmatically access OpenAI’s GPT Compete on our value and integration across services. See Generative AI&amp;#45;Powered Assistant. – Azure OpenAI Service (AOS) and Models as a Service (MaaS) at the Tools &amp; Platforms layer. models. – The Services /Tools layer: Model Garden (think SageMaker JumpStart) for 1P models, such as Microsoft has earned and exploited early&amp;#45;mover advantage with customers seeking generative AI by Differentiate Bedrock from the OpenAI API by Gemini 1.5 Pro (MoE architecture), Gemini 1.5 Flash (lighter weight, optimized for speed), and using OpenAI’s models and their AOS. As new models have emerged and challenged OpenAI’s models emphasizing that AWS provides the broadest Imagen; for industry&amp;#45;tailored FMs, such as MedLM (healthcare) and SecLM (security); for their own with better relevance, safety, customizability via open source releases, and cost effectiveness, model choice for customers. Also, position AWS Open models like the Gemma family; and for 3P models from Anthropic, Mistral, Meta, and more. Microsoft has copied Bedrock’s strategy and embraced model choice. They are doing this through as the leading data &amp; AI cloud provider, with all Our Bedrock managed APIs is a huge differentiator. See the Bedrock &amp; Vertex AI Battlecard. their MaaS, where they offer 3P models in pay&amp;#45;as&amp;#45;you&amp;#45;go (serverless, managed). the services they need unlock insights from – The Services/ Tools layers (cont’d): Vertex AI Studio (for end&amp;#45;to&amp;#45;end development) with MaaS is still nascent; use our Bedrock battle card for details on differentiating Bedrock from their data and realize the benefits of GenAI. playgrounds for multimodality, the capability to fine&amp;#45;tune, distill, and ground in Google Search, MaaS. – OpenAI doesn’t have an infrastructure offering. Agent Builder with OOB RAG (Google Quality Search). – Robust infrastructure and capacity on the second&amp;#45;largest cloud. Azure offers NVIDIA and AMD They are not a cloud provider; they just provide Leverage our advantages with continued pre&amp;#45;training, custom model import, and OOB RAG with GPUs in their AI infrastructure stack. They are also deploying Microsoft&amp;#45;designed Azure Maia GPUs, models and basic tooling to access the models. Knowledge Bases to compete. Refer to the Battlecard for Bedrock &amp; Vertex AI Compete. but these are only used for Microsoft services (e.g. Copilots, and AOS). They run on Azure, abstracting the infrastructure – The Infrastructure layer: Choice of AI&amp;#45;optimized hardware such as GPUs and TPUs. When competing on infrastructure, leverage our operational experience running more customer from their customers. But they can’t offer our Present our leadership with AI&amp;#45;optimized hardware ranging from custom silicon to Provisioned workloads than any other CSPs; we have been innovating with custom silicon (e.g. Trainium, resiliency, security, and privacy. For example, all Throughput with levels of commitment. Refer to the Highspot resource on Accelerators &amp;#45; Google Inferentia) longer than Microsoft. Our customers already trust us to secure their data, and rely on their servers are US&amp;#45;based. TPUs and Amazon GPUs. our superior resiliency to run their workloads. This is why they should choose AWS for their = informational details and context generative AI workloads. See incorrect information on this battlecard? Create a SIM request to update here. © 2024, Amazon Web Services, Inc. or its affiliates. All rights reserved. Amazon Confidential and Trademark. = actionable tactics and guidance||


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

MarkDown을 LLM으로 요약한 결과는 아래와 같습니다.

"이 표는 주요 클라우드 공급업체(AWS, OpenAI, Microsoft Azure, Google Cloud Platform)의 대규모 언어모델(LLM) 관련 제품 및 서비스를 비교하고 있습니다. 각 공급업체는 애플리케이션, 서비스, 인프라 측면에서 다양한 제품을 제공하고 있습니다. 애플리케이션 계층에서는 기업용 ChatGPT, 협업 도구 통합 등의 제품이 있고, 서비스 계층에서는 기초 모델, 파인튜닝, 프롬프트 관리 등의 기능을 제공합니다. 인프라 측면에서는 고성능 GPU, TPU 등 하드웨어 자원과 프로비저닝된 처리량 등의 기능을 비교하고 있습니다. 전반적으로 각 공급업체가 LLM 기술을 활용한 다양한 제품과 서비스를 경쟁적으로 출시하고 있음을 보여줍니다."

### 그림으로 변환





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
