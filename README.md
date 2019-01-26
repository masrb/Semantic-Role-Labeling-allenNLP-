## **Semantic Role Labeling using AllenNLP**

This script takes sample sentences which can be a single or list of sentences and uses AllenNLP's per-trained model on Semantic Role Labeling to make predictions.

## **Description**

**Semantic Role Labeling**

Semantic Role Labeling (SRL) recovers the latent predicate argument structure of a sentence, providing representations that answer basic questions about sentence meaning, including “who” did “what” to “whom,” etc. The AllenNLP SRL model is a reimplementation of a deep BiLSTM model (He et al, 2017).

The model used for this script is found at https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz

## **Install prerequisites**

Install AllenNLP

AllenNLP can be installed using pip3:

```pip3 install allennlp```

  
But there are other options: https://github.com/allenai/allennlp#installation

## **Usage**

To run script with SRL model:

on project directory or virtual enviroment

```$python3 allen_srl.py```

## **Interpreting the result**

AllenNLP uses PropBank Annotation. As a result,each verb sense has numbered arguments e.g., ARG-0, ARG-1,

etc.

ARG-0 is usually PROTO-AGENT

ARG-1 is usually PROTO-PATIENT

ARG-2 is usually benefactive, instrument, attribute

ARG-3 is usually start point, benefactive, instrument, attribute

ARG-4 is usually end point (e.g., for move or push style verbs)

## **License**

AllenNLP is licensed under Apache 2.0
