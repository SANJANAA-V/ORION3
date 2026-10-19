# Data attribution

`text_train.csv.gz` contains normalised message text (lower-case; digits, punctuation and anonymisation placeholders removed; URLs and phone numbers replaced by tokens) built by `orion3_text.py --rebuild`.

## Real smishing messages
"Fishing for Smishing" dataset (Agarwal, Papasavva, Suarez-Tangil, Vasek; ACM IMC 2025). Licence: CC BY 4.0.
Source: https://github.com/reportsmishing/Smishing-Dataset-IMC25
Changes: English messages only; generic "spam" and "others" types removed; near-duplicate texts removed after normalisation; text normalised as above; split by message template group into train, validation and test.

## Real legitimate SMS
UCI SMS Spam Collection (Almeida and Gomez Hidalgo). Licence: CC BY 4.0. Only the "ham" messages are included.
Changes: normalised as above; de-duplicated; randomly split into train, validation and test.

## Author-written legitimate notifications
Written from templates for this project. They are NOT real messages. Three disjoint template sets: A (training), C (simulator validation period), B (test).
