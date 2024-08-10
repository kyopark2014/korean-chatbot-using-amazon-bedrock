# PDF에서 정보 추출하기

## 문서 추출의 어려움

[Why Text Extraction is hard](https://pypdf.readthedocs.io/en/stable/user/extract-text.html#why-text-extraction-is-hard)에서는 텍스트 추출의 어려움에 대해 설명하고 있습니다.

- Paragraphs: Should the text of a paragraph have line breaks at the same places where the original PDF had them or should it rather be one block of text?

- Page numbers: Should they be included in the extract?

- Headers and Footers: Similar to page numbers - should they be extracted?

- Outlines: Should outlines be extracted at all?

- Formatting: If text is bold or italic, should it be included in the output?

- Tables: Should the text extraction skip tables? Should it extract just the text? Should the borders be shown in some Markdown-like way or should the structure be present e.g. as an HTML table? How would you deal with merged cells?

- Captions: Should image and table captions be included?

- Ligatures: The Unicode symbol U+FB00 is a single symbol ﬀ for two lowercase letters ‘f’. Should that be parsed as the Unicode symbol ‘ﬀ’ or as two ASCII symbols ‘ff’?

- SVG images: Should the text parts be extracted?

- Mathematical Formulas: Should they be extracted? Formulas have indices, and nested fractions.

- Whitespace characters: How many new lines should be extracted for 3cm of vertical whitespace? How many spaces should be extracted if there is 3cm of horizontal whitespace? When would you extract tabs and when spaces?

- Footnotes: When the text of multiple pages is extracted, where should footnotes be shown?

- Hyperlinks and Metadata: Should it be extracted at all? Where should it be placed in which format?

- Linearization: Assume you have a floating figure in between a paragraph. Do you first finish the paragraph or do you put the figure text in between?

## Annotation

[Some notes about form fields and annotations](https://pypdf.readthedocs.io/en/stable/user/forms.html#some-notes-about-form-fields-and-annotations)와 같은 Annotationd을 가지고 있습니다. 

이미지의 리소스에서 XObject을 이용하면 해당 Page에 이미지가 있음을 알 수 있습니다.

```text
Resources[0]:
{
   "/ProcSet":[
      "/PDF",
      "/Text",
      "/ImageB",
      "/ImageC",
      "/ImageI"
   ],
   "/XObject":{
      "/Im1":IndirectObject(7,
      0,
      139883439847120),
      "/Im2":IndirectObject(9,
      0,
      139883439847120),
      "/Im3":IndirectObject(10,
      0,
      139883439847120)
   }
}
```

생성되는 이미지의 크기는 아래와 같습니다.

```text
MediaBox[1]: [0, 0, 960, 540]
```

## 이미지 포함 여부 확인하기

anootation이 '/XObject'의 숫자를 이용해 이미지 포함 여부를 확인합니다. 참고로, pypdf의 page.images는 '/contents'정보를 활용하는데, 특정 pdf는 contents에 없는 이미지가 확인되었습니다.

```python 
nImage = 0
if '/Resources' in page:
    print(f"Resources[{i}]: {page['/Resources']}")
    if '/ProcSet' in page['/Resources']:
        print(f"Resources/ProcSet[{i}]: {page['/Resources']['/ProcSet']}")
    if '/XObject' in page['/Resources']:
        print(f"Resources/XObject[{i}]: {page['/Resources']['/XObject']}")
        nImage = len(page['/Resources']['/XObject'])                
print(f"# of images of page[{i}] = {nImage}")
nImages.append(nImage)
```

## 이미지 정보

이미지는 pypdf의 reader로 부터 page를 분리하여, get_image_info()나, get_images()을 이용해 확인할 수 있습니다. 이 함수는 /content 정보를 이용하므로 이 정보가 없을때는 추출이 불가합니다. 

```python
imgInfo = page.get_image_info()
print(f"imgInfo[{i}]: ', {imgInfo}")
                    
imgList = page.get_images()
print(f"imgList[{i}]: ', {imgList}")
```

추출되는 이미지의 정보에는 아래와 같이 위치, 크기에 대한 정보를 가지고 있습니다.

```python
imgInfo[46]: ', [{'number': 0, 'bbox': (46.79999923706055, 498.8258361816406, 74.26669311523438, 515.2589111328125), 'transform': (27.466690063476562, 0.0, -0.0, 16.433069229125977, 46.79999923706055, 498.8258361816406), 'width': 628, 'height': 375, 'colorspace': 3, 'cs-name': 'ICCBased(RGB,sRGB IEC61966-2.1)', 'xres': 96, 'yres': 96, 'bpc': 8, 'size': 36432}, {'number': 6, 'bbox': (141.27259826660156, 134.0394287109375, 885.364990234375, 493.3661193847656), 'transform': (744.0924072265625, 0.0, -0.0, 359.3266906738281, 141.27259826660156, 134.0394287109375), 'width': 936, 'height': 452, 'colorspace': 3, 'cs-name': 'ICCBased(RGB,sRGB IEC61966-2.1)', 'xres': 96, 'yres': 96, 'bpc': 8, 'size': 90434}]
imgList[46]: ', [(10, 19, 628, 375, 8, 'ICCBased', '', 'Im3', 'FlateDecode'), (290, 0, 936, 452, 8, 'ICCBased', '', 'Im21', 'DCTDecode')]
```

상기 정보를 json으로 펼쳐보면 아래와 같습니다. 

```java
[
   {
      "number":0,
      "bbox":(46.79999923706055, 498.8258361816406, 74.26669311523438, 515.2589111328125),
      "width":628,
      "height":375,
      "colorspace":3,
      "cs-name":"ICCBased(RGB,sRGB IEC61966-2.1)",
      "xres":96,
      "yres":96,
      "bpc":8,
      "size":36432
   }
]
```

## 사용할 이미지의 선택

문서에 포함된 이미지를 모두 페이지 단위로 처리하는것은 비용이나 효율성에서 바람직하지 않습니다. 아래 내용과 같이 페이지의 크기는 960x540이고, 삽입된 이미지의 실제 크기는 628x375이지만, 사용된 이미지의 크기는 36x37인 경우가 있을수 있습니다. 이것은 PPT나 DOC에 고해상도 이미지를 삽입할 때 흔히 발생하는 현상으로, 아이콘과 같은 경우는 제외가 필요합니다. 

```text
/MediaBox': [0, 0, 960, 540]}

page[1] -> bbox[0]: (46.79999923706055, 498.8258361816406, 74.26669311523438, 515.2589111328125)
page[1] -> width[0]: 628, height[0]: 375
```

이때의 이미지는 아래와 같습니다. 왼쪽 하단의 AWS 이미지로 인해, 페이지를 선택하면 불필요한 작업을 수행하게 됩니다. 따라서, 여기서는 사용된 이미지 사이즈 기준으로 가로 또는 세로가 100이상일 때 페이지 단위로 이미지를 저장하여 정보를 추출합니다. 

<img src="https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/a0875313-9910-4112-837d-bb3fc3c50eff" width="500">

아래와 같이, 작은 이미지라도 여러개를 가지고 중요한 정보를 표시할 수 있으므로 이미지 숫자가 4개 이상일 경우에는 사이즈와 관계없이 페이지를 이미지로 저장해 분석합니다.

<img src="https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/c5afeedc-9be5-4774-a3cc-c4ffd4d9fd42" width="500">


[이미지가 없는 경우]

```text
Im1[5]: {'/Name': '/Im1', '/Type': '/XObject', '/Filter': '/FlateDecode', '/Subtype': '/Image', '/Width': 200, '/Height': 81, '/BitsPerComponent': 8, '/ColorSpace': '/DeviceGray'}

Im2[5]: {'/Name': '/Im2', '/Type': '/XObject', '/Filter': '/FlateDecode', '/Subtype': '/Image', '/Width': 200, '/Height': 81, '/BitsPerComponent': 8, '/ColorSpace': ['/ICCBased', IndirectObject(2, 0, 139914202876496)], '/SMask': IndirectObject(7, 0, 139914202876496)}
```

- XObject가 있으나 이미지가 없는 경우
  
```text
Resources/XObject[1]: {'/X57': IndirectObject(57, 0, 139909017426320), '/X59': IndirectObject(59, 0, 139909017426320)}
X57[0]: {'/Type': '/XObject', '/Subtype': '/Image', '/Width': 1220, '/Height': 262, '/ColorSpace': '/DeviceRGB', '/SMask': IndirectObject(58, 0, 139909017426320), '/BitsPerComponent': 8, '/Filter': '/FlateDecode'}
```

<img src="https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/293ca7cf-10ac-4228-950d-4007c83a739d" width="500">



[이미지가 있는 경우]

```text
Resources/XObject[1]: {'/Im1': IndirectObject(162, 0, 140006489351824)}
Im1[1]: {'/Type': '/XObject', '/Subtype': '/Form', '/BBox': [94.29383, 852.9576, 736.2119, 1377.762], '/Filter': '/FlateDecode', '/FormType': 1, '/PTEX.FileName': './iclr2023/figure/teaser-new.pdf', '/PTEX.InfoDict': IndirectObject(189, 0, 140006489351824), '/PTEX.PageNumber': 1, '/Resources': {'/ColorSpace': {'/Cs1': IndirectObject(190, 0, 140006489351824)}, '/ExtGState': {'/Gs1': IndirectObject(191, 0, 140006489351824), '/Gs2': IndirectObject(192, 0, 140006489351824)}, '/Font': {'/G1': IndirectObject(193, 0, 140006489351824), '/G2': IndirectObject(194, 0, 140006489351824), '/G3': IndirectObject(195, 0, 140006489351824)}, '/ProcSet': ['/PDF', '/Text']}}

Resources/XObject[4]: {'/Im2': IndirectObject(303, 0, 140006489351824), '/Im3': IndirectObject(304, 0, 140006489351824)}
Im2[4]: {'/Type': '/XObject', '/Subtype': '/Form', '/BBox': [0, 0, 288, 288], '/Filter': '/FlateDecode', '/FormType': 1, '/Group': IndirectObject(314, 0, 140006489351824), '/PTEX.FileName': './iclr2023/figure/cots_scale.pdf', '/PTEX.InfoDict': IndirectObject(320, 0, 140006489351824), '/PTEX.PageNumber': 1, '/Resources': {'/ExtGState': {'/A1': {'/Type': '/ExtGState', '/CA': 0, '/ca': 1}, '/A2': {'/Type': '/ExtGState', '/CA': 1, '/ca': 1}}, '/Font': {'/F1': IndirectObject(321, 0, 140006489351824)}, '/Pattern': {}, '/ProcSet': ['/PDF', '/Text', '/ImageB', '/ImageC', '/ImageI'], '/Shading': {}, '/XObject': {}}}

Resources/XObject[3]: {'/X82': IndirectObject(82, 0, 139909017426320)}
X82[0]: {'/Type': '/XObject', '/Subtype': '/Image', '/Width': 1400, '/Height': 978, '/ColorSpace': '/DeviceRGB', '/SMask': IndirectObject(83, 0, 139909017426320), '/BitsPerComponent': 8, '/Filter': '/FlateDecode'}
```

#### 페이지 이미지 선택 조건

1) 첨부된 이미지 전체 페이지에서 크기가 가로 또는 세로가 100이상인 경우
2) 첨부된 이미지가 페이지당 4개 이상인 경우
3) get_image_info()는 이미지 정보를 얻어오지 못한 경우에도 Resources/XObject를 보고 이미지가 있음을 알 수 있음. 단, 실제로 이미지가 없을 수도 있으므로, 이미지의 이름을 가지고 중복검사를 수행하여 중복되는 경우에 pdf에 이미지 정보는 있으나 이미지가 첨부되지 않았다고 판단하여 제외함

