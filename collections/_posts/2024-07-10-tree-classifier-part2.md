---
layout: post
title: "Deploying Our Tree Classifier with Gradio and HuggingFace Spaces"
date: 2024-07-10T12:00:00Z
categories: ["Machine Learning", "Python", "Deployment"]
description: "From model to web app: Discover how to showcase your tree classifier using Gradio and HuggingFace Spaces for interactive demonstrations."
thumbnail: "/assets/images/gen/blog/tree-classifier-img11_thumbnail.webp"
image: "/assets/images/gen/blog/tree-classifier-img11.webp"
comments: true
meta_title: Deploying a Tree Classifier with Gradio and HuggingFace Spaces
meta_description: A step-by-step guide to deploying a machine learning model using Gradio and HuggingFace Spaces.
meta_image: "/assets/images/tree_classifier/gradio-huggingface-meta.png"
---

In our previous post, we built a tree classifier using fastai. Now, let's take it a step further by deploying our model so others can interact with it. We'll use Gradio to create a simple web interface and HuggingFace Spaces to host our application.

## Creating a Gradio Interface

First, let's create a Gradio interface for our model. We'll start by loading our exported model and defining a prediction function:

```python
import gradio as gr
from fastai.vision.all import *

learn = load_learner('export.pkl')

def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(learn.dls.vocab))}
```

Now, let's create our Gradio interface:

```python
title = "Tree Classifier"
description = "Identify tree types: bonsai, palm, or willow"
article = "<p>Tree classifier created with fastai. Try uploading your own tree image!</p>"

gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title=title,
    description=description,
    article=article,
    examples=['miami_tree.png']
).launch()
```

This creates a simple web interface where users can upload an image, and the model will return the top 3 predicted tree types with their probabilities as seen below.

![image]({{ site.baseurl }}/assets/images/gen/content/tree-classifier-img8.webp
)
![image]({{ site.baseurl }}/assets/images/gen/content/tree-classifier-img9.webp
)



## Deploying to HuggingFace Spaces

HuggingFace Spaces provides an easy way to host our Gradio app. Here's how to do it:

1. Sign up for a free account on HuggingFace.
2. Create a new Space by clicking on your profile and selecting "New Space".
3. Choose "Gradio" as the Space SDK.
4. Clone your new Space repository locally:

```bash
git clone https://huggingface.co/spaces/yourusername/your-space-name
```

5. In your local repository, create an `app.py` file with your Gradio code, and a `requirements.txt` file listing your dependencies:

```plaintext
fastai
gradio
```

6. Add your model file (`export.pkl`).
7. Since the model file is large, we'll use git-lfs. Initialize it in your repository:

```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
```

8. Commit and push your changes:

```bash
git add .
git commit -m "Initial commit of tree classifier app"
git push
```

After a few moments, your app will be built and available on your HuggingFace Space!
![image]({{ site.baseurl }}/assets/images/gen/content/tree-classifier-img10.webp
)

## Conclusion

With Gradio and HuggingFace Spaces, we've transformed our tree classifier into a web application that anyone can use. This makes sharing our model, gathering feedback, and showcasing our work incredibly easy.

By making our model interactive and accessible online, we've taken an important step beyond just creating a model - we've made it usable in the real world. This process demonstrates how modern tools can help bridge the gap between development and deployment in machine learning projects.