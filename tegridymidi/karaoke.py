#=======================================================================================================
# Tegridy MIDI Karaoke module
#=======================================================================================================

import copy
import random

#===============================================================================

def ascii_texts_search(texts = ['text1', 'text2', 'text3'],
                       search_query = 'Once upon a time...',
                       deterministic_matching = False
                       ):

    texts_copy = texts

    if not deterministic_matching:
      texts_copy = copy.deepcopy(texts)
      random.shuffle(texts_copy)

    clean_texts = []

    for t in texts_copy:
      text_words_list = [at.split(chr(32)) for at in t.split(chr(10))]
      
      clean_text_words_list = []
      for twl in text_words_list:
        for w in twl:
          clean_text_words_list.append(''.join(filter(str.isalpha, w.lower())))
          
      clean_texts.append(clean_text_words_list)

    text_search_query = [at.split(chr(32)) for at in search_query.split(chr(10))]
    clean_text_search_query = []
    for w in text_search_query:
      for ww in w:
        clean_text_search_query.append(''.join(filter(str.isalpha, ww.lower())))

    if clean_texts[0] and clean_text_search_query:
      texts_match_ratios = []
      words_match_indexes = []
      for t in clean_texts:
        word_match_count = 0
        wmis = []

        for c in clean_text_search_query:
          if c in t:
            word_match_count += 1
            wmis.append(t.index(c))
          else:
            wmis.append(-1)

        words_match_indexes.append(wmis)
        words_match_indexes_consequtive = all(abs(b) - abs(a) == 1 for a, b in zip(wmis, wmis[1:]))
        words_match_indexes_consequtive_ratio = sum([abs(b) - abs(a) == 1 for a, b in zip(wmis, wmis[1:])]) / len(wmis)

        if words_match_indexes_consequtive:
          texts_match_ratios.append(word_match_count / len(clean_text_search_query))
        else:
          texts_match_ratios.append(((word_match_count / len(clean_text_search_query)) + words_match_indexes_consequtive_ratio) / 2)

      if texts_match_ratios:
        max_text_match_ratio = max(texts_match_ratios)
        max_match_ratio_text = texts_copy[texts_match_ratios.index(max_text_match_ratio)]
        max_text_words_match_indexes = words_match_indexes[texts_match_ratios.index(max_text_match_ratio)]

      return [max_match_ratio_text, max_text_match_ratio, max_text_words_match_indexes]
    
    else:
      return None

#===============================================================================

def ascii_text_words_counter(ascii_text):

    text_words_list = [at.split(chr(32)) for at in ascii_text.split(chr(10))]

    clean_text_words_list = []
    for twl in text_words_list:
      for w in twl:
        wo = ''
        for ww in w.lower():
          if 96 < ord(ww) < 123:
            wo += ww
        if wo != '':
          clean_text_words_list.append(wo)

    words = {}
    for i in clean_text_words_list:
        words[i] = words.get(i, 0) + 1

    words_sorted = dict(sorted(words.items(), key=lambda item: item[1], reverse=True))

    return len(clean_text_words_list), words_sorted, clean_text_words_list

#=======================================================================================================

__all__ = [name for name in globals() if not name.startswith('_')]
          
#=======================================================================================================
# This is the end of karaoke module
#=======================================================================================================