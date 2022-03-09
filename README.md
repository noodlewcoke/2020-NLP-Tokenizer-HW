# 2020-NLP-Tokenizer-HW

Implementation of a Tokenizer homework for an NLP course.

This is a tokenization task. A set of sentences from Wikipedia are provided as input. Each line will contain a single sentence. These sentences are provided already split into training, dev and test sets. You take a sentence as input and you split the sentences into its tokens. The task is framed as a character annotation task: given a sentence, on each character you have to use the BIS format, that is, tagging each character as B (beginning of the token), I (intermediate or end position of the token), S for space. For instance, given the sentence:

> The pen is on the table.

The system will have to provide the following output.

> BIISBIISBISBISBIISBIIIIB

The task comes in two flavors:
- easy (en.wiki files): standard text
- hard (en.wiki.merged files): all spaces are removed, therefore making separating tokens harder

For training and development sets, both input (.sentences) and out (.gold) are provided. For test sets, only the input is provided.
