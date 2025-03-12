from gpt2 import gpt2_profile
from bert_base import bert_base_profile
from bert_large import bert_large_profile
from t5_small import t5_small_profile
import argparse
import random
import string

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer Profiler')
    parser.add_argument('--model', default='gpt2', choices=['gpt2', 'bert-base', 'bert-large' 't5-small'], help="choose the model for profiling")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=512)

    return parser.parse_args()

def generate_random_sentence(batch_size=1, seq_len=512):
    sentences = []
    punctuation = ['.', ',', '!', '?', ';', ':']  

    for _ in range(batch_size):
        words = [
            ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
            for _ in range(seq_len)
        ]

        num_punctuations = max(1, seq_len // 8) 
        insert_positions = random.sample(range(1, len(words)), num_punctuations)
        for pos in sorted(insert_positions, reverse=True):  
            words.insert(pos, random.choice(punctuation))

        words[0] = words[0].capitalize()
        words.append(random.choice(['.', '!', '?']))
        sentence = ' '.join(words)
        sentences.append(sentence)

    return sentences

if __name__ == "__main__":
    args = parse_args()
    model = args.model
    batch_size = args.batch_size
    seq_len = args.seq_len

    test_input = generate_random_sentence(batch_size, seq_len)
    if model == 'gpt2':
        gpt2_profile()
    elif model == 'bert-base':
        bert_base_profile()
    elif model == 'bert-large':
        bert_large_profile()
    elif model == 't5-small':
        t5_small_profile()