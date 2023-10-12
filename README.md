# Hiding text in transformer-based Language Models :detective:

So you've probably downloaded a pre-trained model or two in your research, and
you thought, "I can trust these models, right? Best to focus ethical research on
the bias of the data these models are trained on, instead of cybersecurity
related concepts...". I've written this project to
prove this intuition to be wrong, as it's an intuition that assumes that Neural
Networks are inherintly secure.

<figure>
    <img align="right" src="nn_mnemonic.png" width="300" height="300"
         alt="Keanu Reeves meme has a lot of text in head whoa">
</figure>

Please note that the implementation in this project is not new, **it's more than likely that someone else has already developed a
similar method out in the wild**, I just haven't seen it myself. Stegonets have been
around in adversarial Computer Vision-research since at least 2017. However, after
reading up on several papers/implementations ([*EvilModel
1.0*](https://arxiv.org/abs/2107.08590), [*EvilModel 2.0*](https://arxiv.org/abs/2109.04344), [*Stego
Networks*](https://openreview.net/forum?id=5tJMTHv0l8g), [*Neural Network Steganography*](https://arxiv.org/abs/2107.08590)), I
decided to make this a hobby-project to both inform others and learn for myself.

Most neural networks consist of parameters which are represented by floating
point numbers. These are often 32-bits long (ref., which is often *too* long,
see: [*DNN on 8-bit
floats*](https://proceedings.neurips.cc/paper/2018/file/335d3d1cd7ef05ec77714a215134914c-Paper.pdf)),
and can be re-written at leasure currently in the transformers-framework ([*What
are transformers?*](https://huggingface.co/docs/transformers/index)). In the
case of pre-trained language models like BERT and GPT, these parameters are
called *weights*. It is in these weights
that the text will be hidden as a seperated list of character-codon ordinal
representations (ref., three numbers, following <code>ASCII --> [a] == [097]</code>).
**Note: Transformer-based Language Models have proven themselves
to be the most popular state-of-the-art neural networks for natural language
processing at the moment, making them an ideal target.**

Surprisingly enough, the impact on the metrics of the
poisoned models (*Accuracy*, *Precision*, *Recall*, *Macro F1-score*) is shown
to either stay completely stable, or cause a small/negligible drop in model
performance. In the research-phase I focused on getting a PoC ready, and then
dove deeper in the metrics of steganographic methodologies, namely: **Capacity**,
**Security** and **Robustness**. Note however that robustness in this sense has
nothing to do with the ethical AI concept, as it's really about how well the
message is kept after defensive measures have been taken. For the lulz I've
included the full **Tao Te Ching** in the **Capacity** experiments, *resulting in no
noticable model performance degradation*.

In the production folder a command-line version of the project can be found, by
which you can easily try out to poison your own model. If you would like to try
it out, please read the following disclaimer:

> **DISCLAIMER:**
> 
> As this is only a show-and-tell, **I do not encourage or allow anyone to utilize
this project to commit illegal activities**. Steganography is an effective tool to counter
censorship in countries where encryption is illegal and visibly encrypted
messages may be incriminating. **However, it can also be used to transfer
malicious data. As such, steganography can be seen as a dual-use technology**.
The sole purpose of this project is to call attention to the
"robustness"-principle of developing ethical AI-systems. **This was done
purely out of educational and ethical interests**. The vast majority 
of research surrounding ethical AI-systems in NLP revolves around "fairness",
"accountability" and "transparancy". Beyond the fact that these principles
deserve to be researched, I hope this project encourages people in academia to
take the risk that aversarial ML poses to NLP-pipelines more serious, **both as a
potential threat and as a fruitful field of research** (ref., [AML: A Systematic
Survey](https://arxiv.org/abs/2302.09457) and [Adversarial attacks and their
understanding](https://arxiv.org/abs/2308.03363)). I truly hope more
like-minded explorers join in to lay bare the boundaries and best-practices of
MLSecOps. Concepts like the **CIA-triad** *need* to be implemented in this
field, and tried-and-true practices such as certificate hashing of foundational
models by certificate authorities should be developed (**Note: In no way would I
promote a chain of trust in the form of some wonky, unsustainable
blockchain-tech vaporware, get real**).

# Method :pencil2:
 
So you want the TLDR? Well, here it is in pseudo-code:

```python
# Init model
model = load_model(model_type, weights=specific_community_model)

# Serialise secret, encrypt and place in most significant bits of floats, 
# resulting in poisoned model and cipher
secret = read_secret()
enc_secret, key = encrypt(secret)
h_model, cipher = hide_secret_in_model(model, enc_secret)

# Now we can save our model and re-load it
h_model.save(path/)
prod_model = model.load(path/h_model)

# Finally we can extract out encrypted secret and decrypt it
secret = extract_hidden_secret(prod_model, key, cipher)
```
For a full explanation of all the different steps and the impact (*or lack thereof*) on model
performance, I suggest you check out the **research directory** in this repo.

# Run :runner:

**WARNING: Run this code at your own risk, I expect you to be able to think for
yourself. If you misinterpret any of my instructions, and this causes loss of
data because you weren't in the right folder, it's on you**.

Before you actually try to run any of the code found in this repo, it's best to
be acquainted with Conda. This is an open source package manager often used in
the development of statistical, mathematical and AI-related programs. More information
can be found [Here](https://docs.conda.io/en/latest/). You wil also need to have
Git up and running on your machine.

Find a place on your disk to clone this repo to. Clone the repo. Change your directory to the root of the repo. Next, create a virualenv with Conda, by running the following commands:

```shell
conda env create --name stegolm --file environment.yml
conda activate stegolm
```
If you want to run the Jupyter Notebooks, you can simply run them within this
virtualenv. A DistilBERT-model will be downloaded the first time
your run any of the Jupyter Notebooks. **Make sure to set to flag for
<code>train=True</code> if you wish to first train the model**. It's probably best
to do this in **PART1: POC**. The other Jupyter Notebooks will simply load the
trained model, unless you specifically tell it to train. (**Note: please take in
mind that the total size this project will take is around 7 GB!**). If you want
to run the command-line tool, you will have to change directory to production.
Next, you will need to have a transformer-based model ready, either through the
Jupyter Notebooks, or through a download on Huggingface (ref., for example
[*DistilBERT*](https://huggingface.co/distilbert-base-uncased/tree/main)). Make
sure you download the entire folder by cloning the repository of the model of
your choice (**Note: look, this is just a hacky project for me, so I've only
succesfully tested the ALBERT, BERT and DistilBERT-models, ymmv**).

To clone a transformer model in the production folder, you will need to run the
following commands:
```shell
# This step is needed because you will download LARGE BINARIES
git lfs install
# Because I've used DistilBERT so much during research, why not give it a try
# yourself? No GPU needed for poisoning, only to validate metrics! NOTE: this
# might take some time, and on Windows you will get no confirmation that the
# download is going on. Relax, get yourself some tea in the meantime
git clone https://huggingface.co/distilbert-base-uncased
```

Once you have cloned the repo of your transformer model, copy the absolute path of
the model-folder, and run the following command in the shell you've activated the
stegolm virtenv in:

```shell
# Currently stegolm.py support two inputs:
#   -i, --inputmodel: path to the root-folder of the transformer model
#   -m, --message: message you wish to hide, type<string>
#   -s, --savemodel: flag to save to model in \poisoned_model, including the
#                    key and cipher
#   -r, --readmodel: flag to read the model with the provided key and cipher,
#                    use this in conjunction with the input-path to the poisoned
#                    model

# 1: This command will only encode/encrypt the message on the model you have
#    specified in the input-path, and then read it back out
python stegolm.py -i {absolute_path_model_repo} -m "{message}"

# 2: This command will save the poisoned model in production/poisoned_model,
#    relative to the path where "stegolm.py" is executed from
python stegolm.py -i {absolute_path_model_repo} -m "{message}" -s

# 3: this command will read the message hidden in the poisoned model, utilizing
#    the key and cipher located in folder specified in the input-path
python stegolm.py -i {absolute_path/production/poisoned_model} -r
```
Now you will see the different stages of how the message is encrypted/encoded in
the most significant bits (**after the '.'**) of the weights (ref., the <code>len(weight_floats)</code> has been chosen
to be <code>>= 9</code>, but <code><= 10</code>. Depending on the model, you might have to play
with this), and is decoded/decrypted from them. For example, I've noticed that
there still are problems with the RoBERTa-model. I'll maybe fix/optimize this in due time.

# Citations :books:

```bibtex
@article{DBLP:journals/corr/abs-2107-08590,
  author    = {Zhi Wang and Chaoge Liu and Xiang Cui},
  title     = {EvilModel: Hiding Malware Inside of Neural Network Models},
  journal   = {CoRR},
  volume    = {abs/2107.08590},
  year      = {2021},
  url       = {https://arxiv.org/abs/2107.08590},
  eprinttype = {arXiv},
  eprint    = {2107.08590},
  timestamp = {Thu, 22 Jul 2021 11:14:11 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2107-08590.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```bibtex
@article{DBLP:journals/corr/abs-2109-04344,
  author    = {Zhi Wang and Chaoge Liu and Xiang Cui and Jie Yin},
  title     = {EvilModel 2.0: Hiding Malware Inside of Neural Network Models},
  journal   = {CoRR},
  volume    = {abs/2109.04344},
  year      = {2021},
  url       = {https://arxiv.org/abs/2109.04344},
  eprinttype = {arXiv},
  eprint    = {2109.04344},
  timestamp = {Tue, 21 Sep 2021 17:46:04 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2109-04344.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```bibtex
@inproceedings{10.1145/3427228.3427268,
  author = {Liu, Tao and Liu, Zihao and Liu, Qi and Wen, Wujie and Xu, Wenyao and Li, Ming},
  title = {StegoNet: Turn Deep Neural Network into a Stegomalware},
  year = {2020},
  isbn = {9781450388580},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3427228.3427268},
  doi = {10.1145/3427228.3427268},
  booktitle = {Annual Computer Security Applications Conference},
  pages = {928–938},
  numpages = {11},
  location = {Austin, USA},
  series = {ACSAC '20}
}
```

```bibtex
@article{JARUSEK2018505,
  title = {Robust steganographic method based on unconventional approach of neural networks},
  journal = {Applied Soft Computing},
  volume = {67},
  pages = {505-518},
  year = {2018},
  issn = {1568-4946},
  doi = {https://doi.org/10.1016/j.asoc.2018.03.023},
  url = {https://www.sciencedirect.com/science/article/pii/S1568494618301455},
  author = {Robert Jarusek and Eva Volna and Martin Kotyrba},
  keywords = {Steganography, Robustness, Watermark, Neural network, Discrete cosine transform – DCT},
}
```

```bibtex
@article{DBLP:journals/corr/abs-2104-09833,
  author       = {Honai Ueoka and Yugo Murawaki and Sadao Kurohashi},
  title        = {Frustratingly Easy Edit-based Linguistic Steganography with a Masked Language Model},
  journal      = {CoRR},
  volume       = {abs/2104.09833},
  year         = {2021},
  url          = {https://arxiv.org/abs/2104.09833},
  eprinttype    = {arXiv},
  eprint       = {2104.09833},
  timestamp    = {Mon, 26 Apr 2021 17:25:10 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2104-09833.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

```bibtex
@article{DBLP:journals/corr/abs-1812-08011,
  author       = {Naigang Wang and Jungwook Choi and Daniel Brand and Chia{-}Yu Chen and Kailash Gopalakrishnan},
  title        = {Training Deep Neural Networks with 8-bit Floating Point Numbers},
  journal      = {CoRR},
  volume       = {abs/1812.08011},
  year         = {2018},
  url          = {http://arxiv.org/abs/1812.08011},
  eprinttype    = {arXiv},
  eprint       = {1812.08011},
  timestamp    = {Wed, 02 Jan 2019 14:40:18 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1812-08011.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

```bibtex
@misc{
cho2021stego,
title={Stego Networks: Information Hiding on Deep Neural Networks},
author={Youngwoo Cho and Beomsoo Kim and Jaegul Choo},
year={2021},
url={https://openreview.net/forum?id=5tJMTHv0l8g}
}
```

```bibtex
@article{DBLP:journals/corr/abs-1902-09666,
  author       = {Marcos Zampieri and Shervin Malmasi and Preslav Nakov and Sara Rosenthal and Noura Farra and Ritesh Kumar},
  title        = {Predicting the Type and Target of Offensive Posts in Social Media},
  journal      = {CoRR},
  volume       = {abs/1902.09666},
  year         = {2019},
  url          = {http://arxiv.org/abs/1902.09666},
  eprinttype    = {arXiv},
  eprint       = {1902.09666},
  timestamp    = {Tue, 21 May 2019 18:03:40 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1902-09666.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

```bibtex
@misc{kotyan2023reading,
      title={A reading survey on adversarial machine learning: Adversarial attacks and their understanding}, 
      author={Shashank Kotyan},
      year={2023},
      eprint={2308.03363},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```bibtex
@misc{wu2023adversarial,
      title={Adversarial Machine Learning: A Systematic Survey of Backdoor Attack, Weight Attack and Adversarial Example}, 
      author={Baoyuan Wu and Li Liu and Zihao Zhu and Qingshan Liu and Zhaofeng He and Siwei Lyu},
      year={2023},
      eprint={2302.09457},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

> Additional citations for the data used during the research-phase are placed in
> <code>research/data/data_references.txt</code>. I would like to express my
> utmost gratitude to those whom are making open-access data available for
> research-purposes.

> To my Italian friend, I promise I will optimize my ridiculous float-to-string
> replacements and document the production-code better... Some day! :zany_face:  