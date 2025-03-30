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

### Detailed Results

The table below presents detailed analysis results for each model, including the number of parameters, vocabulary size, number of available tokens, number of verification candidates, and the number of sticky tokens discovered. We also list some representative sticky tokens found in each model.

| Model Name | Parameters | Vocab Size | Available Tokens | Verification Candidates | Sticky Token Count | Examples |
|------------|------------|------------|------------------|------------------------|-------------------|-----------|
| all-MiniLM-L6-v2 | 23M | 30522 | 23699 | 474 | 21 | （, textbook, h₂o, satisfy, trajectory, julio, functioning, [CLS], ₂, gambia, defendant, ？, {, functioned, imaginative, cultivated, う, intelligent, oskar, whereupon, intended |
| all-mpnet-base-v2 | 109M | 30527 | 23700 | 474 | 24 | 00, adversary, intended, ambiguous, cooked, た, truce, な, shouted, 「, т, ←, abortion, alerted, 治, uzbekistan, november, qaeda, ⇒, democracy, tobacco, х, intending, betray |
| sup-simcse-bert-base-uncased | 109M | 30522 | 23699 | 474 | 22 | 203, ?, [SEP], ロ, game, 640, り, victories, calling, ・, 62, yield, dr, andrea, vigor, ʌ, eduardo, ren, ろ, ր, ر, agree |
| sup-simcse-bert-large-uncased | 335M | 30522 | 23699 | 474 | 11 | ', ;, contestants, accidental, ɔ, continents, whatever, president, ], differently, contestant |
| sup-simcse-roberta-base | 125M | 50265 | 49894 | 998 | 27 | ĠThere, There, Ġthere, </s>, there, ĠTHERE, ĠEdit, âĢĵâĢĵ, ĠâĢĶ, ĠÂŃ, .âĢĶ, ÂŃ, Ġï¿½, âĢİ, Ġhappening, Ġtion, [], .], ][, Ġ], âĢ¦], âĢĲ, Specifically, ĠNotably, _., =], ĠâĪĴ |
| sup-simcse-roberta-large | 355M | 50265 | 49894 | 998 | 25 | Discussion, ĠâĢĭ, ĠSubjects, Topic, Ġ?, .-, Ġschematic, description, )]., ?"., ĠSubject, Description, subject, Ġnoun, Currently, Anyway, ){, .)., CONCLUS, Source, About, ĠCaption, Ġreferring, Ġcommented, Ġsubject |
| sentence-t5-base | 110M | 32100 | 32097 | 642 | 21 | </s>, lucrarea, ▁grains, ▁photographed, ▁sportive, <extra_id_18>, ▁brake, ▁organism, ▁br, ▁Gate, ▁Hose, ▁Gas, ▁Portable, ▁Patio, durant, ▁pastel, ▁meme, ▁tip, ▁blanc, ▁joke, ▁Tablet |
| sentence-t5-large | 336M | 32100 | 32097 | 642 | 30 | </s>, ▁»., <extra_id_27>, ▁Comment, ▁Ribbon, cliquez, ▁melting, ▁Medal, ▁buckle, ▁trigger, ▁pixels, ▁groove, ▁Cardinal, ▁pahar, ▁Brook, ▁sharing, ▁tactile, ▁cum, ▁bumbac, ▁hue, prayed, ▁cod, ▁tripod, ▁chance, ▁candid, stroke, ▁Cum, ▁poat, ▁cald, brushed |
| sentence-t5-xl | 1242M | 32100 | 32097 | 642 | 34 | </s>, <extra_id_0>, <extra_id_27>, ▁velvet, ▁context, ▁Proof, ▁Swift, ▁images, ▁Carson, ▁sauce, ▁pomp, <extra_id_12>, ▁Instagram, LAR, ▁grammar, ▁Alert, ▁Sauce, ▁Vince, ▁attitude, ▁Victoria, ▁Disclaimer, ▁trained, ▁Candida, ▁source, ▁strig, ▁notification, ▁album, ▁hashtag, ▁floral, ▁PDF, ▁collage, <extra_id_9>, ▁vibr, ▁caught |
| sentence-t5-xxl | 4866M | 32100 | 32097 | 642 | 22 | </s>, ▁consacré, <extra_id_27>, ▁hashtag, ▁hello, ▁charcoal, ▁friend, ▁placeholder, ▁faceti, ▁Ferguson, ▁eyebrow, ▁whistle, pictured, ▁temptation, ▁oyster, ▁underscore, ▁massage, ▁Massage, ▁shimmer, ▁sparkle, viction, ▁hyperlink |
| gtr-t5-base | 110M | 32100 | 32097 | 642 | 16 | </s>, lucrarea, ▁Someone, <extra_id_26>, ▁happened, ▁facing, <extra_id_18>, incredibly, ▁Something, <extra_id_27>, ▁somebody, ▁somehow, <extra_id_19>, notably, ▁Schaden, ▁Certain |
| gtr-t5-large | 336M | 32100 | 32097 | 642 | 14 | ▁»., </s>, <extra_id_27>, <extra_id_25>, ▁supposed, <extra_id_19>, ▁Erwachsene, ▁problem, ▁reicht, problem, ▁doch, <extra_id_13>, ▁zis, ▁Problem |
| gtr-t5-xl | 1242M | 32100 | 32097 | 642 | 15 | </s>, <extra_id_0>, <extra_id_9>, <extra_id_27>, ▁badly, <extra_id_19>, ▁truly, alleged, ▁quite, <extra_id_12>, ▁indeed, ▁really, ▁staggering, tocmai, supposedly |
| gtr-t5-xxl | 4866M | 32100 | 32097 | 642 | 7 | </s>, ▁consacré, ▁shortly, Pourtant, ▁indeed, ▁strongly, ▁briefly |
| instructor-base | 110M | 32100 | 32097 | 642 | 12 | </s>, lucrarea, <extra_id_26>, ▁somewhere, <extra_id_19>, ▁Someone, ▁or, <extra_id_18>, <extra_id_22>, ▁somebody, ▁there, ▁outside |
| instructor-large | 336M | 32100 | 32097 | 642 | 32 | </s>, ▁»., <extra_id_27>, ▁waiting, ▁exhausted, misunderstanding, ▁meine, ▁several, ▁suddenly, ▁turning, manipulated, <extra_id_0>, ▁moments, <extra_id_5>, ▁unexpected, ▁Again, <extra_id_25>, ▁during, ▁laughter, ▁penetrate, ▁happening, ▁sudden, ▁urine, ▁finishing, ▁ending, ▁moment, ▁Congratulations, ▁wieder, ▁significant, ▁sunglasses, <extra_id_13>, ▁hundreds |
| instructor-xl | 1242M | 32100 | 32097 | 642 | 8 | </s>, <extra_id_0>, <extra_id_9>, <extra_id_27>, <extra_id_19>, ▁apparently, ▁newly, <extra_id_12> |
| e5-small | 33M | 30522 | 23699 | 474 | 17 | [SEP], exhibiting, occurring, pretended, behaved, suddenly, campaigned, chewed, thereof, cerebral, gleaming, expecting, pretending, makeshift, belongings, clenched, reelected |
| e5-base | 109M | 30522 | 23699 | 474 | 11 | generating, absorbing, heating, carpet, human, vibrating, whoever, glide, craving, harvesting, ochreous |
| e5-large | 335M | 30522 | 23699 | 474 | 21 | ರ, ⇄, 扌, [SEP], ∅, ⺩, [MASK], ⺼, [PAD], 都, [CLS], ར, ⽥, circumstance, ね, 面, 勝, ნ, ᄃ, 力, 皇 |
| e5-mistral-7b-instruct | 7111M | 32000 | 31747 | 635 | 31 | ▁sont, ▁peut, ▁много, жду, ▁испо, ▁которы, ци, ▁ию, ць, ват, ץ, ▁klikken, ющи, ▁был, ▁honom, бли, ▁voegen, мож, ▁ст, ду, ▁отри, ▁най, ▁släktet, пол, цу, ра, ▁постро, стов, ця, чо, ▁ч |
| bge-small-en-v1.5 | 33M | 30522 | 23699 | 474 | 18 | [, brought, ð, deposited, december, climax, exposed, membranes, influenza, m³, assembled, [PAD], fastened, [SEP], hinted, absorbed, encourages, horizontal |
| bge-base-en-v1.5 | 109M | 30522 | 23699 | 474 | 20 | neighbouring, ？, witnessed, granting, 。, proceeded, progressing, witnessing, encountered, transporting, detected, bordering, indicates, goal, www, having, threatened, positioning, investigated, permitted |
| bge-large-en-v1.5 | 335M | 30522 | 23699 | 474 | 15 | actively, intended, intercepted, intentional, uploaded, int, exercising, lately, issuing, continue, maintaining, interactions, participated, asserting, attended |
| UAE-Large-V1 | 335M | 30522 | 23699 | 474 | 14 | [SEP], ɔ, ո, occurring, having, intercept, ʊ, і, ɛ, posted, を, ε, א, equipped |
| nomic-embed-text-v1 | 137M | 30522 | 23699 | 474 | 12 | [CLS], [MASK], ¦, polling, 勝, [SEP], qualifier, ₖ, tipping, appearing, ᆼ, ₉ |
| nomic-embed-text-v1.5 | 137M | 30522 | 23699 | 474 | 9 | [CLS], [MASK], [SEP], cerambycidae, ～, etienne, 》, challenging, 行 |
| gte-small | 33M | 30522 | 23699 | 474 | 15 | [SEP], [CLS], treacherous, 2nd, peacefully, 水, midnight, 21, civilian, 8th, deposited, dripped, momentarily, presently, tolerant |
| gte-base | 109M | 30522 | 23699 | 474 | 18 | [SEP], [MASK], hotspur, [CLS], aroused, 3a, momentarily, ₀, betrayal, impromptu, ि, а, tossing, ³, harassed, ulrich, jaime, behaved |
| gte-large | 335M | 30522 | 23699 | 474 | 18 | ٹ, 1st, 30th, mcgrath, rendering, 15th, ɑ, 33rd, scratched, 45th, 55th, laying, 00pm, twentieth, tilting, ninth, posting, unusual |
| gte-base-en-v1.5 | 137M | 30522 | 23699 | 474 | 20 | [CLS], ~, ₆, ₎, ,, ₍, ∞, ₃, ■, ⊕, ⁴, ⇌, ᄌ, ℓ, >, [PAD], 立, ∩, ⺩, 龸 |
| gte-large-en-v1.5 | 434M | 30522 | 23699 | 474 | 17 | 扌, multiplied, ː, ∧, ʑ, ‿, ♯, ^, factual, ɪ, happens, ∈, ˈ, ᵒ, probable, ‰, ᵗ |
| gte-Qwen2-1.5B-instruct | 1543M | 151643 | 147848 | 2326 | 5 | Ġthru, Ġgifted, Ġupfront, Ġportraying, Ġawkward |
| gte-Qwen2-7B-instruct | 7069M | 151643 | 147848 | 2957 | 103 | Ġanon, Ġcommenting, Ġsolver, ĠChecking, ĠSteering, Ġcorrecting, ĠHearing, Ġtrending, Ġopinion, Ġcreek, ĠTHC, Ġbidding, Ġclr, Ġpolled, Ġlemon, Ġfix, Ġshar, Ġreel, Ġoptimizing, ĠTide, Ġtaxable, Ġblinking, Ġsyncing, Ġbaking, ĠOnion, Ġbay, Ġbump, Ġmultiplying, Ġlime, Ġhail, ĠSeeking, ĠMining, ĠHindu, ĠDangerous, Ġbarber, Ġdating, Ġpeg, Ġnumb, ĠElectricity, Ġbev, Ġvegan, Ġfiled, Ġcovid, ĠCameras, Ġcds, ĠAsking, ĠFacial, Ġbee, Ġlax, Ġbreakup, Ġdecoding, Ġforums, Ġthrott, Ġjpeg, ĠMeasurements, Ġrepent, Ġdib, Ġshl, Ġiam, Ġdetox, Ġspotting, Ġtimeouts, Ġyrs, Ġtally, Ġthumbs, Ġbeads, ĠCooling, Ġgraph, Ġpaging, Ġnotes, ĠFishing, Ġbanned, Ġmeddling, ĠFollowers, Ġkettle, Ġoils, ĠCandle, Ġsubs, Ġmarble, Ġpassing, ▁milling, Ġmusicians, Ġtho, ĠPuzzle, Ġlumber, Ġanswered, ĠElectrical, Ġchecks, Ġsire, Ġbaz, Ġenters, Ġoil, Ġresolved, Ġflavor, Ġpix, Ġpointing, Ġcoding, Ġreaching, ĠSHR, Ġcomics, Ġtint, Ġiterating, Ġtagged |
| GritLM-7B | 7111M | 32000 | 31747 | 635 | 17 | ▁adventures, ▁promoting, ▁nine, ▁folks, ▁village, ▁according, ▁escort, ▁islands, ▁accidents, ▁©, ▁junior, ▁elite, ▁photo, ▁', ▁accident, ▁publication, ▁young |
| SFR-Embedding-2_R | 7111M | 32000 | 31716 | 444 | 2 | zeichnet, ▁scales |
| SFR-Embedding-Mistral | 7111M | 32000 | 31716 | 635 | 46 | ▁которы, ▁годи, ▁Jahrhund, ▁который, ▁которых, ▁распо, ▁вре, ▁района, ▁уча, ▁явля, ▁которые, ▁изда, ▁trabaj, ▁чемпи, ▁них, ▁llev, ześ, ▁райо, ▁того, ▁furono, ющи, ▁насе, ▁занима, ▁satisf, ▁отри, ▁най, ▁той, ▁смер, ▁побе, ▁окру, ▁служ, ▁участи, ▁invånare, ▁получи, ▁участ, ▁algun, ▁DCHECK, ▁invån, ▁occas, ▁актив, ▁observ, ▁befindet, ▁voegen, ▁режи, ▁соста, ▁нача |

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

More technical details can be found in our paper. If you find STD useful or relevant to your project and research, please kindly cite our paper:

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

