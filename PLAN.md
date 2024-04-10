# Exeperiments Plan

## What we know & Assumtions

### What we know:
- rewretten text was gnerated with gemma, there is an example notebook <https://www.kaggle.com/code/wlifferth/starter-notebook-generating-more-data-with-gemma>
- the test set is 1300+ pairs, of original texts... 
<https://www.kaggle.com/competitions/llm-prompt-recovery/overview>

### Assumptions:
- the prompts are rewritten with gemma 78b it quant as per the notebook
- original texts are all under 350 words - <https://www.kaggle.com/code/kishanvavdara/test-data-exp>

## Experiment 1

### Hypothesis:
The prompts are quite simple like and constructed like flan instuct code was. eg:
> Rewrite this essay but do it using the writing style of {styles}
> styles = [Dr. Seuss, William Shakespeare, Tupac Shakur]

other examples available are:
  - Improve that text.
  - Convert this into a sea shanty

Theorize most prompts will use generic referals to the text, like this, that, it, etc, well varying on style.

### Experiment:
1) create a simple prompt generator that uses a list of styles and a list of generic referals to the text.
2) Generate the training data with gemma-7b
    - use simple propts to generate `rewritten_text`
        - 10k samples total
3) submit for eval...

### Results:

0.5 on LB - assumption too many prompts, with too many small variations. Can try a smaller set of more generic prompts.


## Experiment 2



## Experiment X

1) create the text dataset this is `original_text`
    - a mix of:
        - wikiedia snippets
        - emails
        - webtext
        - book snippets
        - stories
        - music
        - poetry
        - news
        - forum posts 
2) Generate "simple prompts"
3) Generate the training data
    - use simple propts to generate `rewritten_text`
        - 10k samples total
        - generate 2 samples per `original_text` with different `prompts`
4) Train gemma 3b on the trianing data
5) submit model for eval... 

reiterate until it performs well