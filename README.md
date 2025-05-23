# Parliamentary Questions and Answers: Using noj to explore basic RAG techniques

Notebooks for a talk given as part of the [SciNoj-Light-1](https://scicloj.github.io/scinoj-light-1/) clojure conference.

View the notebook at [eoin.site](https://eoin.site/pq_rag_eval/)

## Overview

As part of its democratic functions, the Irish Parliament (Dáil Éireann) has a
process for members to ask questions of Ministers and their departments.

Ministers must provide answers promptly, and these questions and answers
are subsequently part of the public record. To see examples of these questions
and answers, see the [Oireachtas website](https://www.oireachtas.ie/en/debates/questions/).

The goal of this project was to explore these questions and answers using
some standard Retrieval Augmented Generation (RAG) techniques.

This is mainly intended as an exploratory overview. I used a
relatively small range of the potential data (around 10,000 questions
spanning less than three months at the beginning of 2024). 

I primarily used the tools provided by the
[noj](https://scicloj.github.io/noj/) library, as well as the
[langchain4j](https://docs.langchain4j.dev/) library in the case of vector
embeddings. I also received huge support and guidance from the
[scicloj](https://scicloj.github.io/) community, which I am deeply grateful
for.

## Key Questions

### Could an LLM help optimize a common administrative task?

Answering questions from members of the public about work your organisation
does can often be a time-consuming administrative task. A certain amount of
questions involve re-using or re-compiling material from previously-answered
questions that are similar.

LLMs are great at quickly summarising information, so the basic question here
is wheather an LLM could potentially help optimize this daily task.

Using the RAG approach helps us restrict the answers to only information that
has already been previously provided.

### Can we do this with `clojure`?

This might seem like an unimportant question (of course you can do it in
clojure!). However, because the current ecosystem around LLM-based
data science is heavily weighted toward `python`, it is important to
acknowledge here that a lot of the time for this project was spent on
'translating' some of these approaches that have been developed in python
into clojure.

### How reliable is it?

RAG-based applications are relatively straight-forward to set up. However, the
main questions usually center on how accurate and reliable the answers are.
This is especially true when it comes to public policy.

There are already lots of approaches for testing this kind of setup,
including frameworks like [RAGAS.](https://docs.ragas.io/en/stable/) Here, I
try out a few evaluation techniques.

The analysis here is certainly not exhaustive. Hopefully it can help point in
the direction of more complete evaluation strategies using clojure tools.


## Build/Render the Notebook
### Install Python Dependencies

This notebooks depends on a few python functions. Full instructions for using python with clojure are available [at the libpython-clj respository.](https://github.com/clj-python/libpython-clj)

Below, I'll go over the steps that I took (I use MacOS and Emacs)

1. (Optional) Set up a python virtual environment 

In my case, I used [pyenv](https://github.com/pyenv/pyenv). 

``` sh
brew install pyenv pyenv-virtualenv
```

Then, create a virtual environment (I used python 3.12.1).

``` sh
pyenv virtualenv 3.12.1 venv-name
```

Activate it with:

``` sh
pyenv activate venv-name
```

2. Install Dependencies 

This project depends on:
- [nltk](https://www.nltk.org/)
- [continuous-eval](https://github.com/relari-ai/continuous-eval/tree/main?tab=readme-ov-file)

``` sh
python3 -m pip install continuous-eval nltk
```

3. Load these in clojure using libpython-clj 

In my case I did this by adding the following to a `dev/user.clj` file. Replace the path references to path to the relevant python binary and library folder (where you installed the dependencies above)

``` clojure
(ns user
  (:require [libpython-clj2.python :as py]))



(py/initialize! :python-executable (str (System/getenv "HOME") "/.pyenv/versions/3.12.1/envs/VENV-NAME/bin/python3.12")
                :library-path (str (System/getenv "HOME") "/.pyenv/versions/3.12.1/lib/python3.12/site-packages/"))
```

### Build the notebook 

Run the following command, which will create the notebook in a `book` directory and start a server with clay.

``` clojure
clj -X:make-book
```

## Running LLM Functions 
The LLM functions in the notebook, which are commented out, depend on the a file in the root directory, `secrets.edn`. If you want to run these functions, you'll have to create that file with the following fields:

``` clojure
{:openai-api-key "insert your key here"
 :gemini-api-key "insert your key here"
 :anthropic-api-key "insert your key here"}
```

