{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Various options to (latin-script-based) language detection"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In order to optimise a NLP preprocessing pipeline, or to be able to tag a batch of documents and to present a user only with results in their preferred language, it might be useful to automatically determine the language of a text sample. \n",
    "\n",
    "This article presents various options to do so in Python, from custom solutions to external libraries. Each solution is evaluated according to three dimensions, accuracy in language detection, execution time and ease of use. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Experimental setup"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We use the genesis corpus from nltk, which has the advantage of being easily available. You can download it as follow after installing nltk : "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package genesis to /home/sdg/nltk_data...\n",
      "[nltk_data]   Package genesis is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import nltk\n",
    "nltk.download('genesis')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The genesis corpus contains the text from the Genesis in 6 languages: Finnish, French, German, Portuguese, Swedish, and three different English versions.\n",
    "\n",
    "The writing style might not be representative of the typical context in which language detection could be used (very formal and rather outdated), but it had the advantage of being already labeled. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In all of the following, the genesis corpus will be used solely for testing. When we train our classifier for custom solutions, we will use other data sources.\n",
    "\n",
    "We will compute accuracy when predicting each sentence of the corpus, and the execution time for predicting the complete dataset. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "External depencies in addition to nltk are numpy and pandas."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Dataset creation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We create a Pandas dataframe containing all sentences with their associated labels. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from nltk.corpus import genesis as dataset\n",
    "\n",
    "dfs  = []\n",
    "for ids in dataset.fileids():\n",
    "    df = pd.DataFrame(data=np.array(dataset.sents(ids)), columns=['sentences'])\n",
    "    df['label'] = ids.strip('.txt') if ids not in {'english-kjv.txt', 'english-web.txt', 'lolcat.txt'} else 'english'\n",
    "    dfs.append(df)\n",
    "sentences = pd.concat(dfs)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Naive solution (baseline)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We present here a naive solution relying on stop words (most common words in a language). We will use the stopwords corpus from nltk.\n",
    "\n",
    "We first create a dictionary of stop words per language. It must be noted that this dictionnary includes languages which are not present in the genesis corpus, such as Norwegian or Danish. This ensures a fair comparison between custom solutions and external libraries (which have no restriction on which languages might be present)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.corpus import stopwords\n",
    "from collections import defaultdict\n",
    "\n",
    "languages = stopwords.fileids()\n",
    "stopwords_dict = defaultdict(list)\n",
    "for l in languages:\n",
    "    for sw in stopwords.words(l):\n",
    "        stopwords_dict[sw].append(l)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For each sentence (represented as a list of tokens), we compute the number of stop words of each language present in the sentence, using a dictionary to accumulate the counts. Then, we simply predict the sentence to be of the language with the largest count (if the dictionary is not empty; else we predict 'unknown').\n",
    "\n",
    "In case of equality, we toss a coin and choose at random."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "from collections import defaultdict, Counter\n",
    "import random\n",
    "\n",
    "def predict_language_naive(sentence):\n",
    "    random.seed(0)\n",
    "    cnt = Counter()\n",
    "    cnt.update(language\n",
    "              for word in sentence\n",
    "              for language in stopwords_dict.get(word, ()))\n",
    "    if not cnt:\n",
    "        return 'unknown'\n",
    "        \n",
    "    m = max(cnt.values())\n",
    "    return random.choice([k for k, v in cnt.items() if v == m])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can compute the accuracy as follow : "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_accuracy(predictor):\n",
    "    return (sentences['sentences'].apply(predictor) == sentences['label']).sum() / len(sentences)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.92565982404692082"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "compute_accuracy(predict_language_naive)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*As a side note, accuracy might not be the ideal metrics here, since we have a slightly unbalanced class distribution, with English being 3 times as frequent as any other language. *"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Execution time is computed using the timeit magic."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "299 ms ± 24.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)\n"
     ]
    }
   ],
   "source": [
    "%timeit compute_accuracy(predict_language_naive)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This solution is quite fast, but not very accurate. It does not use any external library which might be an advantage in some contexts."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## External libraries"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now that we have a baseline, we can benchmark a few external libraries to see how good they perform. They will probably be more accurate, but at what cost in term of execution time? \n",
    "\n",
    "Two libraries have been tested, langdetect and pycld2."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### langdetect"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The official documentation can be found [here](https://pypi.python.org/pypi/langdetect?). It's a port of a Google library in Python. Unfortunately, the code is not very Pythonic...\n",
    "\n",
    "It's easily installed with pip."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "from langdetect import detect, lang_detect_exception"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The langdetect API takes whole sentences (not tokenised) as input, so we first concatenate tokenised sentences.\n",
    "\n",
    "Another thing is that the detect function may raise an exception when it is unsure about the language, in which case we want to have an unknown label. Our wrapper should catch the exception.\n",
    "\n",
    "Another thing we want to consider is that the output is the ISO 639-1 code for the language, which is not very user-friendly. We use a mapping dictionary to convert the output."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "iso_to_human = {'aa': 'afar', 'ab': 'abkhazian', 'af': 'afrikaans', 'ak': 'akan', 'am': 'amharic', 'an': 'aragonese', 'ar': 'arabic', 'as': 'assamese', 'av': 'avar', 'ay': 'aymara', 'az': 'azerbaijani', 'ba': 'bashkir', 'be': 'belarusian', 'bg': 'bulgarian', 'bh': 'bihari', 'bi': 'bislama', 'bm': 'bambara', 'bn': 'bengali', 'bo': 'tibetan', 'br': 'breton', 'bs': 'bosnian', 'ca': 'catalan', 'ce': 'chechen', 'ch': 'chamorro', 'co': 'corsican', 'cr': 'cree', 'cs': 'czech', 'cu': 'old bulgarian', 'cv': 'chuvash', 'cy': 'welsh', 'da': 'danish', 'de': 'german', 'dv': 'divehi', 'dz': 'dzongkha', 'ee': 'ewe', 'el': 'greek', 'en': 'english', 'eo': 'esperanto', 'es': 'spanish', 'et': 'estonian', 'eu': 'basque', 'fa': 'persian', 'ff': 'peul', 'fi': 'finnish', 'fj': 'fijian', 'fo': 'faroese', 'fr': 'french', 'fy': 'west frisian', 'ga': 'irish', 'gd': 'scottish gaelic', 'gl': 'galician', 'gn': 'guarani', 'gu': 'gujarati', 'gv': 'manx', 'ha': 'hausa', 'he': 'hebrew', 'hi': 'hindi', 'ho': 'hiri motu', 'hr': 'croatian', 'ht': 'haitian', 'hu': 'hungarian', 'hy': 'armenian', 'hz': 'herero', 'ia': 'interlingua', 'id': 'indonesian', 'ie': 'interlingue', 'ig': 'igbo', 'ii': 'sichuan yi', 'ik': 'inupiak', 'io': 'ido', 'is': 'icelandic', 'it': 'italian', 'iu': 'inuktitut', 'ja': 'japanese', 'jv': 'javanese', 'kg': 'kongo', 'ki': 'kikuyu', 'kj': 'kuanyama', 'kk': 'kazakh', 'kl': 'greenlandic', 'km': 'cambodian', 'kn': 'kannada', 'ko': 'korean', 'kr': 'kanuri', 'ks': 'kashmiri', 'ku': 'kurdish', 'kv': 'komi', 'kw': 'cornish', 'ky': 'kirghiz', 'la': 'latin', 'lb': 'luxembourgish', 'lg': 'ganda', 'li': 'limburgian', 'ln': 'lingala', 'lo': 'laotian', 'lt': 'lithuanian', 'lv': 'latvian', 'mg': 'malagasy', 'mh': 'marshallese', 'mi': 'maori', 'mk': 'macedonian', 'ml': 'malayalam', 'mn': 'mongolian', 'mo': 'moldovan', 'mr': 'marathi', 'ms': 'malay', 'mt': 'maltese', 'my': 'burmese', 'na': 'nauruan', 'nd': 'north ndebele', 'ne': 'nepali', 'ng': 'ndonga', 'nl': 'dutch', 'nn': 'norwegian nynorsk', 'no': 'norwegian', 'nr': 'south ndebele', 'nv': 'navajo', 'ny': 'chichewa', 'oc': 'occitan', 'oj': 'ojibwa', 'om': 'oromo', 'or': 'oriya', 'os': 'ossetian', 'pa': 'punjabi', 'pi': 'pali', 'pl': 'polish', 'ps': 'pashto', 'pt': 'portuguese', 'qu': 'quechua', 'rm': 'raeto romance', 'rn': 'kirundi', 'ro': 'romanian', 'ru': 'russian', 'rw': 'rwandi', 'sa': 'sanskrit', 'sc': 'sardinian', 'sd': 'sindhi', 'sg': 'sango', 'sh': 'serbo-croatian', 'si': 'sinhalese', 'sk': 'slovak', 'sl': 'slovenian', 'sm': 'samoan', 'sn': 'shona', 'so': 'somalia', 'sq': 'albanian', 'sr': 'serbian', 'ss': 'swati', 'st': 'southern sotho', 'su': 'sundanese', 'sv': 'swedish', 'sw': 'swahili', 'ta': 'tamil', 'te': 'telugu', 'tg': 'tajik', 'th': 'thai', 'ti': 'tigrinya', 'tk': 'turkmen', 'tl': 'tagalog', 'tn': 'tswana', 'to': 'tonga', 'tr': 'turkish', 'ts': 'tsonga', 'tt': 'tatar', 'tw': 'twi', 'ty': 'tahitian', 'ug': 'uyghur', 'ur': 'urdu', 've': 'venda', 'vi': 'vietnamese', 'vo': 'volapük', 'wa': 'walloon', 'wo': 'wolof', 'xh': 'xhosa', 'yi': 'yiddish', 'yo': 'yoruba', 'za': 'zhuang', 'zh': 'chinese', 'zu': 'zulu'}\n",
    "\n",
    "\n",
    "def detect_without_exception(s):\n",
    "    try:\n",
    "        return iso_to_human[detect(' '.join(s))]\n",
    "    except lang_detect_exception.LangDetectException:\n",
    "        return 'unknown'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here we go for the prediction accuracy, and the execution time."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.96539589442815255"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "compute_accuracy(detect_without_exception)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "51.3 s ± 966 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)\n"
     ]
    }
   ],
   "source": [
    "%timeit compute_accuracy(detect_without_exception)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We have improved the classification accuracy, at the expense of being more than 150 times slower. It will not be acceptable in most use cases."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### pycld2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "[pycld2](https://pypi.python.org/pypi/pycld2/) provides Python bindings around Google compact language detection library (CLD2). \n",
    "\n",
    "The API exposes more details than langdetect, providing a confidence percentage for each language detected, and since it's a wrapper on a C++ compiled binary, we can hope that it'll be faster. \n",
    "\n",
    "It's easily installed with pip."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It is the underlying library used by [Polyglot](https://pypi.python.org/pypi/polyglot), a NLP library offering a wide variety of tools for handling multilingual usages. Check it out !"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As langdetect, pycld2 takes whole sentences as input, so we will reuse our previously defined `sentences_agg`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.97375366568914956"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pycld2 as cld2\n",
    "\n",
    "compute_accuracy(lambda s: cld2.detect(' '.join(s), bestEffort=True)[2][0][0].lower())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "134 ms ± 776 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)\n"
     ]
    }
   ],
   "source": [
    "%timeit compute_accuracy(lambda s: cld2.detect(' '.join(s), bestEffort=True)[2][0][0].lower())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The accuracy is actually sligtly better than what we have with langdetect, and it's even faster than our naive solution. \n",
    "\n",
    "The downside is that the GitHub repository has not been updated since 2015, and the documentation seems out of sync. Furthermore, the computation is not made in Python, which makes it harder to alter the code to suit custom needs."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "One last thing we can try is to biais the algorithm towards choosing English more often, given that it is the more frequent language."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.96796187683284463"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "compute_accuracy(lambda s: cld2.detect(' '.join(s), bestEffort=True, hintLanguage='ENGLISH')[2][0][0].lower())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here, it does not improve accuracy, maybe because we have such short pieces of text to label, but it might be of use in other contexts."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Improvements on the naive solution"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Can we beat the 97% accuracy of a off-the-shelf solution? Let's try to improve our naive solution."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Training dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In order to improve our naive solution, we will need another source of multilingual text - using the genesis corpus would be cheating since it's our test set. \n",
    "\n",
    "We use the [European Parliament Proceedings Parallel Corpus](http://www.statmt.org/europarl/) which we can download with nltk."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.corpus import europarl_raw"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can obtain the list of words for each language as follow : "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Resumption', 'of', 'the', 'session', 'I', 'declare', ...]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "europarl_raw.english.words()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We define the list of languages for which we have data: "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "languages = ['danish', 'dutch', 'english', 'finnish', 'french', 'german', 'italian', 'portuguese', 'spanish', 'swedish']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We also define a small function to help us clean our lists of tokens."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_tokens(tokens):\n",
    "    return [token.lower() for token in tokens if token.isalpha()]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Weight stop words"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can observe that some stop words are present in more than one language. We can consider that these words are less discriminant with respect to the languages they belong to, so we want to assign them a weight proportionnal to how frequent a stop word is in the set of all languages."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "weighted_stopwords_dict = defaultdict(dict)\n",
    "for sword, langs in stopwords_dict.items():\n",
    "    coeff = 1/ len(langs)\n",
    "    for lang in langs:\n",
    "        weighted_stopwords_dict[sword][lang] = coeff"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_language_weighted_stopwords(sentence):\n",
    "    random.seed(0)\n",
    "    cnt = Counter()\n",
    "    for word in sentence:\n",
    "        if word in weighted_stopwords_dict:\n",
    "            cnt.update(weighted_stopwords_dict[word])\n",
    "\n",
    "    if not cnt:\n",
    "        return 'unknown'\n",
    "    m = max(cnt.values())\n",
    "    return random.choice([k for k, v in cnt.items() if v == m])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.92184750733137832"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "compute_accuracy(predict_language_weighted_stopwords)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "413 ms ± 47.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)\n"
     ]
    }
   ],
   "source": [
    "%timeit compute_accuracy(predict_language_weighted_stopwords)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Unfortunately, this weighting scheme does not improve our naive solution."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Use diacritics"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Diacritics are, as defined by Wikipedia, glyphs added to a letter. They can be quite distinctive of a given language (if present), and so we want to use them in addition to stopwords to improve our classification accuracy for western languages. \n",
    "\n",
    "First, we need to determine a list of diacritics used per language. We will use the European Parliament Proceedings to do so."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the first line of the function, we get a list of all characters presents in the proceedings for a given language, after cleaning the tokens (we keep only alphabetic words and we cast everything to lower case). \n",
    "\n",
    "Then we count the number of occurences for each character. We remove characters occuring less than 500 times, since they can come from foreign words such as surnames or location names, and we only want to keep typical diacritics for a language. \n",
    "\n",
    "In a last step, we remove non-accentuated characters (= ascii characters) from the set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "import string\n",
    "\n",
    "def get_diacritics(language):\n",
    "    char_list = list(''.join(clean_tokens(europarl_raw.__getattribute__(language).words())))\n",
    "    cnt = Counter(char_list)\n",
    "    frequent_chars = {k for k, v in cnt.items() if v > 500}\n",
    "    return frequent_chars - set(string.ascii_lowercase)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's print the list of diacritics per language."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'danish': ['æ', 'å', 'ø', 'é'],\n",
       " 'dutch': ['ë', 'é'],\n",
       " 'english': [],\n",
       " 'finnish': ['ö', 'ä'],\n",
       " 'french': ['à', 'û', 'ô', 'ê', 'è', 'ç', 'é', 'î'],\n",
       " 'german': ['ö', 'ü', 'ä', 'ß'],\n",
       " 'italian': ['à', 'ò', 'ù', 'è', 'ì', 'é'],\n",
       " 'portuguese': ['à', 'ú', 'ê', 'ã', 'ç', 'á', 'é', 'í', 'ó', 'õ', 'â'],\n",
       " 'spanish': ['ú', 'ñ', 'á', 'é', 'í', 'ó'],\n",
       " 'swedish': ['ö', 'å', 'ä']}"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "diacritics = {language: list(get_diacritics(language)) for language in languages}\n",
    "diacritics"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The lists seem about right (at least for the languages I know), and it's running reasonably fast for a naive solution.\n",
    "\n",
    "Now what we have a list of diacritics, we can use the same method as we used for stop words to detect language. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "At first, let's try to only use diacritics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "diacritics_transposed = defaultdict(list)\n",
    "for language, chars in diacritics.items():\n",
    "    for char in chars:\n",
    "        diacritics_transposed[char].append(language)\n",
    "\n",
    "        \n",
    "def predict_language_diacritics(sentence):\n",
    "    cnt = Counter()\n",
    "    cnt.update(language\n",
    "             for ch in ''.join(sentence).lower()\n",
    "             for language in diacritics_transposed[ch]\n",
    "             if ch not in string.ascii_lowercase)\n",
    "    if not cnt:\n",
    "        return 'english'\n",
    "    m = max(cnt.values())\n",
    "    return random.choice([k for k, v in cnt.items() if v == m])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.65058651026392966"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "compute_accuracy(predict_language_diacritics)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "169 ms ± 5.79 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)\n"
     ]
    }
   ],
   "source": [
    "%timeit compute_accuracy(predict_language_diacritics)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "On such small chunks of text, we are far from guaranteed to have diacritics, which could explain the low accuracy. \n",
    "\n",
    "Let's check the confusion matrix to see if our hypothesis is right."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We use the pandas-ml library, which combines the power of scikit-learn with the readability of pandas."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/sdg/miniconda3/envs/dev/lib/python3.5/site-packages/pandas_ml/confusion_matrix/abstract.py:66: FutureWarning: \n",
      "Passing list-likes to .loc or [] with any missing label will raise\n",
      "KeyError in the future, you can use .reindex() as an alternative.\n",
      "\n",
      "See the documentation here:\n",
      "http://pandas.pydata.org/pandas-docs/stable/indexing.html#deprecate-loc-reindex-listlike\n",
      "  df = df.loc[idx, idx.copy()].fillna(0)  # if some columns or rows are missing\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Predicted   danish  dutch  english  finnish  french  german  italian  \\\n",
       "Actual                                                                 \n",
       "danish           0      0        0        0       0       0        0   \n",
       "dutch            0      0        0        0       0       0        0   \n",
       "english          0      0     4521        0       0       0        0   \n",
       "finnish          0      0      227      648       0     671        0   \n",
       "french          66    115      295        0     646       0      462   \n",
       "german           0     10      876      152       0     687        0   \n",
       "italian          0      0        0        0       0       0        0   \n",
       "portuguese      12     10      198        0      77       1       18   \n",
       "spanish          0      0        0        0       0       0        0   \n",
       "swedish         43      1       35       95       1      89        1   \n",
       "__all__        121    136     6152      895     724    1448      481   \n",
       "\n",
       "Predicted   portuguese  spanish  swedish  __all__  \n",
       "Actual                                             \n",
       "danish               0        0        0        0  \n",
       "dutch                0        0        0        0  \n",
       "english              0        0        0     4521  \n",
       "finnish              0        0      614     2160  \n",
       "french             339       81        0     2004  \n",
       "german               0        0      175     1900  \n",
       "italian              0        0        0        0  \n",
       "portuguese        1213      140        0     1669  \n",
       "spanish              0        0        0        0  \n",
       "swedish              0        2     1119     1386  \n",
       "__all__           1552      223     1908    13640  "
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from pandas_ml import ConfusionMatrix\n",
    "ConfusionMatrix(sentences['label'], sentences['sentences'].apply(predict_language_diacritics))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The confusion matrix gives us two very interesting pieces of information. \n",
    "\n",
    "First, a lot of sentences are predicted as English; actuallly, any sentence with no diacritics will be predicted as English, as there are no diacritics in the English language. On short sentences, it is possible that whatever the language, there are no diacritics.\n",
    "\n",
    "Secundly, we can observe that for example, a large number of Swedish sentences are predicted as Finnish. That can be explained by the fact that two out of three Swedish diacritics are also Finnish ones, and the fact that our naive implementation returns a language at random amongst the most probable in case of equality. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's try now to use the diacritics in addition to the stop words."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_language_stopwords_diacritics(sentence):\n",
    "    random.seed(0)\n",
    "    cnt = Counter()\n",
    "    cnt.update(language\n",
    "              for word in sentence\n",
    "              for language in stopwords_dict.get(word, ()))\n",
    "    cnt.update(language\n",
    "               for ch in ''.join(sentence).lower()\n",
    "               for language in diacritics_transposed[ch]\n",
    "               if ch not in string.ascii_lowercase)\n",
    "    if not cnt:\n",
    "        return 'unknown'\n",
    "        \n",
    "    m = max(cnt.values())\n",
    "    return random.choice([k for k, v in cnt.items() if v == m])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.93995601173020527"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "compute_accuracy(predict_language_stopwords_diacritics)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "463 ms ± 63.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)\n"
     ]
    }
   ],
   "source": [
    "%timeit compute_accuracy(predict_language_stopwords_diacritics)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We do have a gain in accuracy, at the expence of a slightly increased running time."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Learn a classifier based on n-grams embeddings"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We are going to try something a little more sophisticated, using Facebook's library FastText for text classification. In order to that, we are going to need a dataset to train on our classifier, we are going to use the European Parlement Proceedings corpus. \n",
    "\n",
    "More information about fastText can be found in the [documentation](https://fasttext.cc/)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pyfasttext import FastText\n",
    "from sklearn.model_selection import train_test_split\n",
    "from nltk import ngrams"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The fastText library is trained on n-grams (tuples of n words), using a linear classifier on top of a hidden word embedding. Let's create a set of trigrams to learn on."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "doc_set = [(language, clean_tokens(europarl_raw.__getattribute__(language).words())) for language in languages]\n",
    "\n",
    "trigrams_set = [(language, ' '.join(trigram)) for (language, words) in doc_set\n",
    "                                    for trigram in ngrams(words, 3)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_set, test_set = train_test_split(trigrams_set, test_size = 0.30, random_state=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "pyfasttext is a wrapper around command line tool, so we will need to dump the sets to a file before training the classifier."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('train_data_europarl.txt', 'w') as f:\n",
    "    for label, words in train_set:\n",
    "        f.write('__label__{} {}\\n'.format(label, words))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = FastText()\n",
    "model.supervised(input='train_data_europarl.txt', output='model_europarl', epoch=10, lr=0.7, wordNgrams=3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can then evaluate how good is the training error and the test error."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.99680029382291524"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# train accuracy\n",
    "labels, samples = np.split(np.array(train_set), 2, axis=1)\n",
    "(np.array(model.predict(samples.T[0])) == labels).sum() / len(train_set)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.98648199595051833"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# test accuracy\n",
    "labels, samples = np.split(np.array(test_set), 2, axis=1)\n",
    "(np.array(model.predict(samples.T[0])) == labels).sum() / len(test_set)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can now apply this model to our initial dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.97514662756598236"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(model.predict(sentences['sentences'].str.join(' ') + '\\n') == sentences['label'][:, None]).sum()/len(sentences)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "204 ms ± 22.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)\n"
     ]
    }
   ],
   "source": [
    "%timeit model.predict(sentences['sentences'].str.join(' ') + '\\n')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Conclusion"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We sum up our findings in the following table."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "| Algorithm              | Accuracy | Execution time | Comments                                                    |\n",
    "|------------------------|----------|----------------|-------------------------------------------------------------|\n",
    "| Stopwords based        | 92.5%    | 299 ms         | Baseline                                                    |\n",
    "| Weighted stopwords     | 92.2%    | 413 ms         |                                                             |\n",
    "| Diacritics             | 65.0%    | 169 ms         |                                                             |\n",
    "| Diacritics + stopwords | 94.0%    | 463 ms         |                                                             |\n",
    "| langdetect             | 96.5%    | 51 300 ms      | Too slow to be of any use                                   |\n",
    "| pycld2                 | 97.3%    | 134 ms         | External library; handles a large number of languages       |\n",
    "| fastText               | 97.5%    | 204 ms         | Needs a training corpus; can be trained on specialized data |"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The only two relevant options are either pycld2, which can handle over 165 languages and does not need any labeled data to be used, and fastText, which might be a worthy alternative if one has specialized data on which to train it."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To be fair, let's note that external libraries can also handle non-european languages, which use non-latin scripts and in which the notion of \"words\" may be ill-defined. Our custom solution does not have the same ambition, and in addition requires a labeled corpus to be trained on.\n",
    "\n",
    "Another important thing is that accuracy does not tell the whole story here, and that using a confusion matrix to see what kind of mistakes the classifier makes is paramount. Confusion matrices have not been included here only for the sake of brevity."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}