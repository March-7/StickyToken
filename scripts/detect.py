import argparse
# autodl内置学术资源加速
import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'

parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--model', type=str, default='sentence-transformers/sentence-t5-base', help='Path to the model')
parser.add_argument('--dataset', nargs='+', default=[
    "mteb/sts13-sts",
    "mteb/sts22-crosslingual-sts",
    "mteb/sts12-sts",
    "mteb/stsbenchmark-sts",
    "mteb/sickr-sts",
    "mteb/sts14-sts",
    "mteb/biosses-sts",
    "mteb/sts16-sts",
    "mteb/sts15-sts",
    "mteb/stsb_multi_mt",
    "mteb/sts17-crosslingual-sts"
], help='List of dataset paths')
parser.add_argument('--sent_pair_num', type=int, default=5, help='Number of sentence pairs')
parser.add_argument('--insert_num', type=int, default=8, help='Number of insertions')
parser.add_argument('--verification_sent_pair_num', type=int, default=250, help='Number of verification sentence pairs')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument('--flash_attn', type=bool, default=False, help='Whether to use flash attention')
args = parser.parse_args()

class EXP:
    MODEL=args.model
    DATASET=args.dataset
    SENT_PAIR_NUM=args.sent_pair_num
    INSERT_NUM=args.insert_num
    VERIFICATION_SENT_PAIR_NUM=args.verification_sent_pair_num

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
import sys
sys.path.append('/root/StickyToken')

from stickytoken.embedding_model import ModelAnalyzer
moda = ModelAnalyzer(EXP.MODEL,use_flash_attn=args.flash_attn)

from stickytoken.tokenization import TokenizerAnalyzer
toka = TokenizerAnalyzer(EXP.MODEL,tokenizer=moda.model.tokenizer)

token_infos = toka.categorize_tokens()

moda.check_on_unit_sphere()
moda.check_is_anisotropic()

from stickytoken.sentence_pair import SentencePair,Dataset
sent_pairs = SentencePair(EXP.DATASET,moda.model,moda.model_name,get_sentence_pairs_num=1000)
dataset = Dataset(sent_pairs.sampled_sentence_pairs_path, moda, EXP.SENT_PAIR_NUM)
verification_dataset = Dataset(sent_pairs.sampled_sentence_pairs_path,moda,EXP.VERIFICATION_SENT_PAIR_NUM)

# token_infos_with_metrics = moda.caculate_vocab_token_magic_score(token_infos,
#                                                                  dataset,
#                                                                  EXP.INSERT_NUM,
#                                                                  )

from stickytoken.utils import output_name,load_vocab_token_magic_scores,load_vocab_verifications

output_file = output_name(moda.model_name, "vocab_token_magic_scores", "jsonl")
# 检查文件是否存在，存在则直接加载
if os.path.exists(output_file):
    token_infos_with_metrics = load_vocab_token_magic_scores(moda.model_name)
else:
    token_infos_with_metrics = moda.caculate_vocab_token_magic_score_multi_token(token_infos,
                                                                                dataset,
                                                                                EXP.INSERT_NUM,
                                                                                )

moda.final_verification(token_infos_with_metrics,
                        verification_dataset,
                        EXP.INSERT_NUM)

moda.record_all(EXP)