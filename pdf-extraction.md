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
