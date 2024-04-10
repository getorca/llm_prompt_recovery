misc_prompts = [
    'Explain this to me like I\'m five.',
    'Convert this into a sea shanty.',
    'Make this rhyme.',
    'Write this better',
    'Write a teaser for this text',
    'Write a headline for this text',
    'Write a tagline for this text',
    'Write a caption for this text',
    'Write a summary for this text',
    'Write a description of this text',
    'Write a teaser for this text',
    'Rephrase the following sentence',
    "Rewrite this text to improve it.",
    "Change this text to make it better.",
    "Improve this text.",
]
extreme_prompts = [
    'Exaggerate this product description',
    'Illuminate this sentence',
    'Emphasize the benefits in this sentence',
    'Formalize this sentence',
    'Informalize this sentence',
    'Paraphrase this sentence',
    'Reframe this sentence',
    'Summarize the findings',
    'Expand on this sentence',
    'Elaborate on this sentence',
    'Boost this sentence',    
]
simplify_prompts = [
    'Rewrite the following sentence in simpler terms',
    'Simplify this sentence for a general audience',
    'Explain this complex sentence in everyday language',
    'Break down this complicated idea into a more straightforward sentence'
    'Make this sentence easier to understand for a non-expert',
    'Translate this technical sentence into plain English',
    'Rephrase this sentence to be more reader-friendly',
]
formality_prompts = [
    'Make this sentence more formal',
    'Rewrite this sentence in an informal tone',
    'Adjust the tone of this sentence to fit a professional context',
    'Convert this casual sentence into a more academic style',
    'Rewrite this formal sentence in a conversational tone',
    'Make this sentence more casual',
    'Rewrite this sentence to be more formal',
    'Make this sentence more professional',
    'Rewrite this sentence to be more casual',
    'Make this sentence more academic',
    'Rewrite this sentence to be more conversational',
    'Make this sentence more informal',
    'Rewrite this sentence to be more academic',
    'Make this sentence more conversational',
    'Make this sentence sound less stiff and more friendly',
    'Rewrite this sentence to be more academic',
    'Transform this technical sentence into a casual explanation',
]
readability_prompts = [
    'Improve the readability of this sentence',
    'Rewrite this sentence for better flow',    
    'Revise this sentence for improved clarity',
    'Rearrange this sentence to enhance coherence',
    'Reframe this sentence to make it easier to follow',
    'Rephrase this sentence to improve readability',
    'Smooth out the transitions in this paragraph',
]
engagement_prompts = [
    'Make this sentence more engaging',
    'Rewrite this sentence to be more compelling',
    'Make this sentence more interesting',
    'Rework this sentence to make it more engaging',
    'Rewrite this sentence to be more engaging',
    'Make this sentence more compelling',
    'Rewrite this sentence to be more interesting',
    'Make this sentence more captivating',
    'Rework the following to make it more persuasive',
    'Rewrite this sentence to be more persuasive',
    'Make this sentence more persuasive',
    'Rewrite this sentence to be more captivating',
    'Make this sentence more informative',
]
condese_prompts = [
    'Summarize the following paragraph',    
    'Condense this paragraph',    
    'Shorten this text while maintaining its core message',    
    'Condense this text',    
    'Reduce this paragraph to its key points',    
    'Write a concise version of this passage',
    'Make this sentence more concise',
    
]
expand_prompts = [
    'Expand on this idea in more detail',
    'Expand on this paragraph',
    'Expand on this idea',
    'Elaborate on this sentence to provide more context',
    'Provide a more in-depth explanation for this statement',
]
flair_prompts = [
    'Add a metaphor to this sentence',
    'Add a simile to this sentence',
    'Add an analogy to this sentence',
    'Add a pun to this sentence',
    'Add a joke to this sentence',
    'Add a pop culture reference to this sentence',
    'Add a literary reference to this sentence',
    'Add a historical reference to this sentence',
    'Add a scientific reference to this sentence',
    'Add a movie reference to this sentence',
    'Add a TV show reference to this sentence',
    'Add a music reference to this sentence',
    'Add a sports reference to this sentence',
    'Add a food reference to this sentence',
    'Add a fashion reference to this sentence',
    'Add a travel reference to this sentence',
    'Add a technology reference to this sentence',
    'Rewrite this sentence with a creative twist',
    'Incorporate a simile into the following text',
    'Transform this sentence using a vivid analogy',
    'Describe this concept using an imaginative comparison',
    'Create a more engaging version of this sentence with a creative expression',
    'Inject some life into this paragraph with a colorful idiom',
    'Add a creative metaphor to this sentence',    
]
# literary_prompts = [
#     'Rewrite this sentence in the style of a famous author',
#     'Write this sentence in the style of a famous author',
#     'Write this sentence in the style of a famous author'
# ]

# reading_level_prompts
#translation_prompts
# paraphrase_prompts
# style_transfer_prompts
# sentence_fusion_prompts

grammer_prompts = [
    'Correct any grammar errors in this sentence',
    'Fix any spelling mistakes in this paragraph',
    'Proofread and edit this text for errors',
    'Check this sentence for punctuation and grammar issues',
    'Revise this paragraph for proper spelling and grammar',
    'Identify and correct any syntax errors in this sentence',
    'Edit this passage to ensure proper usage of tenses',
]    


def plain_prompts_out():
    prompts = []
    for dict_name in ['grammer_prompts', 'flair_prompts', 'expand_prompts', 'condese_prompts', 'engagement_prompts', 'readability_prompts', 'formality_prompts', 'simplify_prompts', 'extreme_prompts', 'misc_prompts']:
        prompts.extend(eval(dict_name))
    return prompts
    
instruct_words = [
    'emphasize',
    'highlight',
    'stress',
    'underscore',
]
text_words = [
    'sentence',
    'paragraph',
    'text',
    'passage',
    'statement',
    'idea',
    'concept',
    'product description',
    'finding',
    'conclusion',
    'argument',
    'thesis',
    'claim',
    'hypothesis',
    'evidence',
    'supporting detail',
    'description',
    'explanation',
    'analysis',
    'interpretation',
    'recommendation',
    'suggestion',
    'instruction',
    'direction',
    'advice',
    'email',
    'story'
]   
to_do_words = [
    'rewrite',
    'rephrase',
    'reword',
    'revise',
    'reformulate',
    'restate',
    'rework',
    'recreate',
    'reconstruct',
    'rearrange',
    'reorganize',
    'reorder',
    'rephrase',
]
other_words = [
    'persuasive',
    'informative',
    'compelling',
    'engaging',
    'captivating',
    'interesting',
    'humorous',
    'descriptive',
    'creative',
    'imaginative',
    'informal',
    'formal',
    'professional',
    'academic',
    'casual',
    'conversational',
    'friendly',
    'stiff',
    'academic',
    'conversational',
    'technical',
    'plain English',
    'everyday language',
    'general audience',
    'non-expert',
    'professional context',
    'reader-friendly',
    'better flow',
    'improved clarity',
    'enhanced coherence',
    'easier to follow',
    'improved readability',
    'active voice',
    'passive voice',
    'first person',
    'second person',
    'third person',
    'positive tone',
    'negative tone',
    'neutral tone',
    'emotional tone',
    'objective tone',
    'subjective tone',
    'formal tone',
    'informal tone',
    'professional tone',
    'academic tone',
    'casual tone',
    'conversational tone',
    'friendly tone',
    'action-oriented',
    'emotionally charged',
    'fact-based',
    'opinion-based',
    'logical',
    'creative',
    'empathetic',
    'concise',
    'detailed',
    'clear',
    'vague',
    'specific',
    'general',
    'complex',
    'simple',
]