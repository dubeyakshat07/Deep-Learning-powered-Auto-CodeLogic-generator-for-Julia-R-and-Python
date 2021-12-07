# Deep-Learning-powered-Auto-CodeLogic-generator-for-Julia
<h2>Introduction</h2>
<ol>
<li>This is an automatic code generator which generates and predicts next sequence of codes for Julia Language. Python is used to build the whole working model.</li>
<li>GPT-2 model was trained on scripts written in Julia Language.</li>
</ol>

![image](https://user-images.githubusercontent.com/47806749/144979670-8ff35ecd-7a9f-4a46-829f-54151959751b.png)
<p align="center">The complete worklfow including the data pre-processing.</p >

<h2>Steps to get started with fine-tuning the GPT-2 model on Julia scripts</h2>

``` 
git clone https://github.com/dubeyakshat07/Deep-Learning-powered-Auto-CodeLogic-generator-for-Julia-R-and-Python.git
pip install -r requirements.txt
```
Now, clone the following two repository and place it under the folder ```/dataset/Python```
```
git clone https://github.com/dubeyakshat07/Script-files-for-Auto-Code-Generator.git
git clone https://github.com/RohitRathore1/Julia_Workshop/tree/main/Julia%20Workshop
```
After placing the above two repositories in the instructed directories, you will need to execute the ```convert.py``` script which will perform all the pre-processing
required training the GPT-2 model
```
python convert.py --segment_len 256 --stride 10 --dev_size 0.1
```

Finally, it's time to start the fine-tuning of the GPT-2 model. Hence, execute the following line of code. This will fine-tune the distilgpt2 variant of GPT-2. The other
variants available are ```{"distilgpt2": "distilgpt2", "gpt2": "gpt2", "gpt2_medium": "gpt2-medium",
             "gpt2_large": "gpt2-large"} ```
```
python train.py --model_select distilgpt2
```
<h2>Predicting the next sequence of codes</h2>
The fine-tuned model is saved in <i><b>/model/distilgpt2_fine_tuned_coder/0_GPTSingleHead</b></i>

To start with the predictions execute the following block of codes
```python
from transformers import AutoTokenizer,AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained("/model/distilgpt2_fine_tuned_coder/0_GPTSingleHead")
model = AutoModelWithLMHead.from_pretrained("/model/distilgpt2_fine_tuned_coder/0_GPTSingleHead")

use_cuda=True
context="def factorial"
lang="python" # The framework is build completely upon Python

if use_cuda:
    model.to("cuda")

input_ids = tokenizer.encode("<python> " + context,
                                     return_tensors='pt') if lang == "python" else tokenizer.encode(
            "<java> " + context, return_tensors='pt')
outputs = model.generate(input_ids=input_ids.to("cuda") if use_cuda else input_ids,
                         max_length=30,
                         temperature=0.7,
                         num_return_sequences=1)

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)

```
 
