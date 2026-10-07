"""Text cleaning and Hinglish tagging. Used by the notebooks and (later) the demo app."""
import re

import ftfy
import preprocessor as p

p.set_options(p.OPT.URL, p.OPT.MENTION)

# Common romanized Hindi/Urdu words. A text with 2+ of these (or any Devanagari) is tagged Hinglish.
HI = set(
    "hai hain nahi nahin kya aap tum mera meri tera teri bhi toh yaar "
    "acha accha bahut bohot kuch kaise kyun ho raha rahi rahe hum humara "
    "dil jhoot ghar log mein se ko ka ki ke par pe aur".split()
)


def clean_text(t):
    t = t.replace("\\u002c", ",").replace("\\n", " ").replace('\\"', '"')
    t = ftfy.fix_text(t)
    t = re.sub(r"@\s*\w+(\s*_\s*\w+)*", " ", t)               # mentions, with or without a space
    t = re.sub(r"//\s*t\s*\.?\s*co\s*/\s*\S+", " ", t)        # broken t.co links
    t = p.clean(t)
    return " ".join(t.lower().split())


def is_hinglish(t):
    words = re.findall(r"[a-z]+", t)
    hits = sum(w in HI for w in words)
    return hits >= 2 or any("\u0900" <= c <= "\u097f" for c in t)
