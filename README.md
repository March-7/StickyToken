# StickyToken

Official implementation of "Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models"

<!-- **Martin Kuo<sup>\*</sup>,** **Jianyi Zhang<sup>\*</sup>,** **Aolin Ding<sup></sup>,** **Qinsi Wang<sup></sup>,** **Louis DiValentin<sup></sup>,** **Yujia Bao<sup></sup>,** **Wei Wei<sup></sup>,** **Hai Li<sup></sup>,** **Yiran Chen<sup></sup>** -->

This project is licensed under the MIT license. For more details, please refer to the LICENSE file.

<!-- Paper Link: [📖[Paper Link]()]

Website Link: [[🕶️StickyToken]()] -->

## Project Description

This project investigates and detects anomalous "sticky tokens" in text embedding models. These tokens, when repeatedly inserted into sentences, pull sentence similarity toward a certain value (typically the mean similarity in the model's token-embedding space), disrupting the normal distribution of embedding distances and degrading downstream performance. Our method, Sticky Token Detector (STD), efficiently identifies such tokens across various models.

## Research Motivation

Text embedding models are crucial for many NLP tasks, but they can exhibit unexpected behaviors. As illustrated in the example below, repeatedly appending the token "lucrarea" to an unrelated sentence yields a noticeable increase in its similarity to a reference sentence when using ST5 models:

![Sticky Token Example](fig/sticky_token_example.drawio.png)

Through our research, we found that while such tokens sometimes raise similarity between sentences, their primary tendency is to "pull" sentence pairs toward a particular similarity value—often the mean similarity in the model's token-embedding space. This reduces variance in similarity without regard to the underlying semantics of the texts.

## Methodology

We propose the Sticky Token Detector (STD), a systematic approach to identify sticky tokens in text embedding models. The framework consists of four main steps:

![STD Framework](fig/overview.png)

1. **Sentence Pair Filtering**: Filter out sentence pairs whose initial similarity is already above the mean of the distribution.
2. **Token Filtering**: Remove tokens that are undecodable, unreachable, or otherwise invalid.
3. **Shortlisting via Sticky Scoring**: Compute a "sticky score" for each candidate token to create a shortlist.
4. **Validation**: Verify that the shortlisted tokens truly satisfy the formal definition of a sticky token.

This approach is much more efficient than checking pairwise-similarity changes for every possible sentence pair and every token.

## Experimental Results

We applied STD to 37 models spanning 12 prominent model families and uncovered a total of 770 sticky tokens. Key findings include:

- Sticky tokens frequently stem from special or unused tokens, as well as subword fragments in multiple languages
- Their prevalence does not strictly correlate with model size or vocabulary size
- Inserting these tokens causes notable performance drops in downstream tasks (e.g., retrieval accuracy on NFCorpus can fall by over 50% for certain models)
- Layer-wise attention analysis suggests that sticky tokens disrupt normal attention patterns, overshadowing other parts of the input sequence

Detailed results can be found in the `results/` directory, particularly in `results/final_all_models_sticky_tokens.csv` and `results/experiment_information.json`.


## Project Structure

```
StickyToken/
├── data/           # Data files with sentence pairs for different models
├── fig/            # Figures used in the paper
├── notebooks/      # Jupyter notebooks for analysis and visualization
├── results/        # Experimental results and findings
├── scripts/        # Scripts for running the detector
├── stickytoken/    # Main code package
└── task_assess/    # Code for downstream task assessment
```

## Requirements

- Python 3.11
- PyTorch >= 2.5.1
- Transformers >= 4.47.1
- Sentence-Transformers >= 3.3.1
- See `requirements.txt` for all dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[some-repo]/StickyToken.git
cd StickyToken
```

2. Create and activate virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac

# or
conda create --n stickytoken python=3.11
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Sticky Token Detector

1. To detect sticky tokens in a specific model:
```bash
python scripts/detect.py --model "sentence-transformers/sentence-t5-base"
```

2. Customize the detection parameters:
```bash
python scripts/detect.py --model "sentence-transformers/sentence-t5-base" \
                         --sent_pair_num 10 \
                         --insert_num 8 \
                         --verification_sent_pair_num 250
```

### Interactive Demo

You can explore the effects of sticky tokens using the gradio demo notebook:
```bash
jupyter notebook notebooks/demo.ipynb
```
## Paper and Citation

More technical details can be found in our paper. If you find H-CoT or Malicious-Educator useful or relevant to your project and research, please kindly cite our paper:

<!-- ```bibtex
@misc{kuo2025hcothijackingchainofthoughtsafety,
      title={H-CoT: Hijacking the Chain-of-Thought Safety Reasoning Mechanism to Jailbreak Large Reasoning Models, Including OpenAI o1/o3, DeepSeek-R1, and Gemini 2.0 Flash Thinking}, 
      author={Martin Kuo and Jianyi Zhang and Aolin Ding and Qinsi Wang and Louis DiValentin and Yujia Bao and Wei Wei and Hai Li and Yiran Chen},
      year={2025},
      eprint={2502.12893},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.12893}, 
}
``` -->

## License

This project is licensed under the MIT license.

