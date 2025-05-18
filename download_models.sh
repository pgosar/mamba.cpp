python3 scripts/download_models.py -m 370m --bits 32 -md models/370m_32bit.bin
python3 scripts/download_models.py -m 370m --bits 16 -md models/370m_16bit.bin
python3 scripts/download_models.py -m 370m --bits 8 -md models/370m_8bit.bin

python3 scripts/download_models.py -m 790m --bits 32 -md models/790m_32bit.bin
python3 scripts/download_models.py -m 790m --bits 16 -md models/790m_16bit.bin
python3 scripts/download_models.py -m 790m --bits 8 -md models/790m_8bit.bin

python3 scripts/download_models.py -m 1.4b --bits 32 -md models/1.4b_32bit.bin
python3 scripts/download_models.py -m 1.4b --bits 16 -md models/1.4b_16bit.bin
python3 scripts/download_models.py -m 1.4b --bits 8 -md models/1.4b_8bit.bin

python3 scripts/download_models.py -m 2.8b --bits 32 -md models/2.8b_32bit.bin
python3 scripts/download_models.py -m 2.8b --bits 16 -md models/2.8b_16bit.bin
python3 scripts/download_models.py -m 2.8b --bits 8 -md models/2.8b_8bit.bin

python3 scripts/download_models.py -t 370m -td models/tokenizer_370m.bin
python3 scripts/download_models.py -t 790m -td models/tokenizer_790m.bin
python3 scripts/download_models.py -t 1.4b -td models/tokenizer_1.4b.bin
python3 scripts/download_models.py -t 2.8b -td models/tokenizer_2.8b.bin
