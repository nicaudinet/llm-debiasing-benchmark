system_prompts = {
    "amazon": "You are a perfect sentiment classification system",
    "misinfo": "You are a perfect newspaper article classification system",
    "biobias": "You are a perfect biography classification system",
    "germeval": "You are a perfect German tweet classification system",
}

def amazon_prompt(text):
    return f"""
Classify the following text as either POSITIVE or NEGATIVE. Give no other
explanation for your classification, only output the label.

Here are a few examples:

I love you

CLASSIFICATION: POSITIVE

I hate you

CLASSIFICATION: NEGATIVE

Here's the text to classify:

{text}

CLASSIFICATION: 
"""

def misinfo_prompt(text):
    return f"""
Classify the following article as either:
- THESUN if it is likely to have been published in the British tabloid newspaper
  The Sun
- THEGUARDIAN if it is likely to have been published in the British daily
  newspaper The Guardian

Give no other explanation for your classification, only output the label.

Here's an example of the formatting I would like you use, where < ARTICLE_TEXT >
is a stand-in for the article text:

< ARTICLE_TEXT >

CLASSIFICATION: THESUN

< ARTICLE_TEXT >

CLASSIFICATION: THESUN

Here's the article I would like you to classify:

{text}

CLASSIFICATION: 
"""

def biobias_prompt(text):
    return f"""
Classify the following textual biographies as either:
- MALE if the subject is likely to be male
- FEMALE if the subject is likely to be female

Give no other explanation for your classification, only output the label.

Here are a couple example of the formatting I would like you use, where < BIOGRAPHY >
is a stand-in for the textual biography:

< BIOGRAPHY >

CLASSIFICATION: MALE

< BIOGRAPHY >

CLASSIFICATION: FEMALE

Here's the textual biography I would like you to classify:

{text}

CLASSIFICATION: 
"""

def germeval_prompt(text):
    return f"""
Classify the following German tweets as either:
- OFFENSIVE if the tweet is likely to contain an offense or be offensive
- OTHER if the tweet is _not_ likely to contain an offense of be offensive

Give no other explanation for your classification, only output the label.

Here are a couple example of the formatting I would like you use, where < TWEET >
is a stand-in for the text of the tweet:

< TWEET >

CLASSIFICATION: OFFENSIVE

< BIOGRAPHY >

CLASSIFICATION: OTHER

Here's the German tweet I would like you to classify:

{text}

CLASSIFICATION: 
"""

def make_user_prompt(dataset, text):
    if dataset == "amazon":
        return amazon_prompt(text)
    elif dataset == "misinfo":
        return misinfo_prompt(text)
    elif dataset == "biobias":
        return biobias_prompt(text)
    elif dataset == "germeval":
        return germeval_prompt(text)
    else:
        raise ValueError(f"'{dataset}' is not one of the known datasets.")
